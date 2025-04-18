import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from safetensors import safe_open
from safetensors.flax import save
import os

# Define your LSTM model using the built-in nn.LSTM
class LSTMModel(nn.Module):
    num_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, carry):
        # h, c = carry
        lstm_out, new_carry = nn.OptimizedLSTMCell(self.num_classes)(x, carry)
        return lstm_out, new_carry

    def initialize_carry(self, batch_size):
        h = jnp.zeros((batch_size, self.hidden_size))
        c = jnp.zeros((batch_size, self.hidden_size))
        return (h, c)

# Define your loss functions
def supervised_loss_fn(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()

def consistency_loss_fn(logits1, logits2):
    probs1 = nn.softmax(logits1)
    probs2 = nn.softmax(logits2)
    return jnp.mean((probs1 - probs2)**2)

# Define your training step
def train_step(state, labeled_batch, unlabeled_batch, consistency_weight=0.1):
    def loss_fn(params):
        (labeled_inputs, labeled_labels), _ = labeled_batch
        initial_carry_labeled = model.apply(
            {"params": params}, jnp.zeros((1, 1, labeled_inputs.shape[-1])), method=model.initialize_carry, rngs=state.rng
        )
        logits_labeled, _ = model.apply({"params": params}, labeled_inputs, initial_carry_labeled, mutable=False)
        supervised_loss = supervised_loss_fn(logits_labeled, labeled_labels)

        (unlabeled_inputs, _), _ = unlabeled_batch
        rng1 = jax.random.fold_in(state.rng, jnp.uint32(1))
        rng2 = jax.random.fold_in(state.rng, jnp.uint32(2))

        initial_carry_unlabeled1 = model.apply(
            {"params": params}, jnp.zeros((1, 1, unlabeled_inputs.shape[-1])), method=model.initialize_carry, rngs={"dropout": rng1}
        )
        logits_unlabeled1, _ = model.apply({"params": params}, unlabeled_inputs, initial_carry_unlabeled1, mutable={"dropout": True}, rngs={"dropout": rng1})

        initial_carry_unlabeled2 = model.apply(
            {"params": params}, jnp.zeros((1, 1, unlabeled_inputs.shape[-1])), method=model.initialize_carry, rngs={"dropout": rng2}
        )
        logits_unlabeled2, _ = model.apply({"params": params}, unlabeled_inputs, initial_carry_unlabeled2, mutable={"dropout": True}, rngs={"dropout": rng2})

        consistency_loss = consistency_loss_fn(logits_unlabeled1, logits_unlabeled2)

        total_loss = supervised_loss + consistency_weight * consistency_loss
        return total_loss, (supervised_loss, consistency_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (supervised_loss, consistency_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss, supervised_loss, consistency_loss

# Define your evaluation step
def eval_step(state, test_batch):
    inputs, labels = test_batch
    initial_carry = model.apply(
        {"params": state.params}, jnp.zeros((1, 1, inputs.shape[-1])), method=model.initialize_carry, rngs=state.rng
    )
    logits, _ = model.apply({"params": state.params}, inputs, initial_carry, mutable=False)
    loss = supervised_loss_fn(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)
    return loss, accuracy

# --- Model Loading and Saving using safetensors ---
MODEL_SAVE_PATH = "flax_lstm_semi_supervised.safetensors"

def save_model(state, path):
    save(path, {"params": state.params})
    print(f"Model saved to: {path}")

def load_model(path):
    with safe_open(path, framework="flax") as f:
        params = f.get_tensor("params")
    print(f"Model loaded from: {path}")
    return state.replace(params=params)

# --- Initialize or Load Model ---
key = jax.random.PRNGKey(0)
num_classes = 10
hidden_size = 64
sequence_length = 20
input_dim = 32
learning_rate = 0.001
num_epochs = 5  # Reduced for demonstration

model = LSTMModel(num_classes=num_classes, hidden_size=hidden_size)
dummy_input = jnp.zeros((1, sequence_length, input_dim))
initial_carry = model.initialize_carry(2)

params = model.init(key, dummy_input, initial_carry)['params']
tx = optax.adam(learning_rate=learning_rate)
initial_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx, rng=key)

if os.path.exists(MODEL_SAVE_PATH):
    state = load_model(MODEL_SAVE_PATH)
else:
    print("No saved model found. Creating and training a new one.")
    state = initial_state

    # Dummy data loaders
def create_dummy_dataloader(num_samples, seq_len, in_dim, num_cls):
    inputs = jax.random.uniform(key, (num_samples, seq_len, in_dim))
    labels = jax.random.randint(key, (num_samples,), 0, num_cls)        
    return [(inputs[i:i+1], labels[i:i+1]) for i in range(0, num_samples)]

labeled_dataloader = create_dummy_dataloader(100, sequence_length, input_dim, num_classes)
unlabeled_dataloader = create_dummy_dataloader(200, sequence_length, input_dim, num_classes)

# Training loop (only if no saved model)
for epoch in range(num_epochs):
    for (labeled_inputs, labeled_labels), (unlabeled_inputs, _) in zip(labeled_dataloader, unlabeled_dataloader):
        labeled_batch = ((labeled_inputs, labeled_labels), None)
        unlabeled_batch = ((unlabeled_inputs, None), None)
        state, total_loss, supervised_loss, consistency_loss = train_step(state, labeled_batch, unlabeled_batch)
        print(f"Epoch: {epoch}, Total Loss: {total_loss:.4f}, SLoss: {supervised_loss:.4f}, CLoss: {consistency_loss:.4f}")

# Save the newly trained model
save_model(state, MODEL_SAVE_PATH)

# --- Evaluation using the loaded or trained model ---
test_dataloader = create_dummy_dataloader(50, sequence_length, input_dim, num_classes)
all_test_losses = []
all_test_accuracies = []
for test_inputs, test_labels in test_dataloader:
    test_loss, test_accuracy = eval_step(state, (test_inputs, test_labels))
    all_test_losses.append(test_loss)
    all_test_accuracies.append(test_accuracy)

mean_test_loss = jnp.mean(jnp.array(all_test_losses))
mean_test_accuracy = jnp.mean(jnp.array(all_test_accuracies))
print(f"Test Loss: {mean_test_loss:.4f}, Test Accuracy: {mean_test_accuracy:.4f}")
