# Load required packages
library(stringr)

# ----------------------------
# RNG class (simplified)
random_u32 <- local({
  state <- 1337
  function() {
    state <<- bitwXor(state, bitwShiftR(state, 12))
    state <<- bitwXor(state, bitwShiftL(state, 25))
    state <<- bitwXor(state, bitwShiftR(state, 27))
    ((state * 0x2545F4914F6CDD1D) %/% 2^32) %% 2^32
  }
})

random_float <- function() {
  (bitwShiftR(random_u32(), 8)) / 16777216.0
}

# ----------------------------
# Data loading and tokenization

train_text <- readLines("data/train.txt")
train_text <- paste(train_text, collapse = "\n")
stopifnot(all(strsplit(train_text, "")[[1]] %in% c("\n", letters)))

uchars <- sort(unique(strsplit(train_text, "")[[1]]))
char_to_token <- setNames(seq_along(uchars) - 1, uchars)
token_to_char <- setNames(uchars, seq_along(uchars) - 1)
EOT_TOKEN <- char_to_token["\n"]
vocab_size <- length(uchars)

tokenize <- function(text) {
  unlist(lapply(strsplit(text, "")[[1]], function(c) char_to_token[[c]]))
}

train_tokens <- tokenize(train_text)
val_tokens <- tokenize(readLines("data/val.txt") |> paste(collapse = "\n"))
test_tokens <- tokenize(readLines("data/test.txt") |> paste(collapse = "\n"))

# ----------------------------
# Data loader
dataloader <- function(tokens, window_size) {
  lapply(1:(length(tokens) - window_size + 1),
         function(i) tokens[i:(i + window_size - 1)])
}

# ----------------------------
# n-gram model
NgramModel <- function(vocab_size, seq_len, smoothing = 0.0) {
  counts <- array(0, dim = rep(vocab_size, seq_len))
  uniform <- rep(1 / vocab_size, vocab_size)
  
  train <- function(tape) {
    stopifnot(length(tape) == seq_len)
    idx <- as.list(tape)
    counts[do.call(`[`, c(list(counts), idx))] <<- counts[do.call(`[`, c(list(counts), idx))] + 1
  }
  
  get_probs <- function(tape) {
    stopifnot(length(tape) == seq_len - 1)
    idx <- as.list(tape)
    slice <- do.call(`[`, c(list(counts), idx))
    counts_vec <- as.numeric(slice) + smoothing
    if (sum(counts_vec) == 0) {
      return(uniform)
    }
    counts_vec / sum(counts_vec)
  }
  
  list(train = train, get_probs = get_probs, counts = counts)
}

# ----------------------------
# Evaluation
eval_split <- function(model, tokens, seq_len) {
  loss <- 0
  count <- 0
  for (tape in dataloader(tokens, seq_len)) {
    x <- tape[1:(length(tape) - 1)]
    y <- tape[[length(tape)]]
    probs <- model$get_probs(x)
    prob <- probs[y + 1]
    loss <- loss - log(prob)
    count <- count + 1
  }
  if (count == 0) return(0)
  loss / count
}

# ----------------------------
# Grid search
seq_lens <- c(3, 4, 5)
smoothings <- c(0.03, 0.1, 0.3, 1.0)
best_loss <- Inf
best_kwargs <- list()

for (seq_len in seq_lens) {
  for (smoothing in smoothings) {
    model <- NgramModel(vocab_size, seq_len, smoothing)
    for (tape in dataloader(train_tokens, seq_len)) {
      model$train(tape)
    }
    train_loss <- eval_split(model, train_tokens, seq_len)
    val_loss <- eval_split(model, val_tokens, seq_len)
    cat(sprintf("seq_len %d | smoothing %.2f | train_loss %.4f | val_loss %.4f\n",
                seq_len, smoothing, train_loss, val_loss))
    if (val_loss < best_loss) {
      best_loss <- val_loss
      best_kwargs <- list(seq_len = seq_len, smoothing = smoothing)
    }
  }
}

# ----------------------------
# Final model training
cat("Best hyperparameters:\n")
print(best_kwargs)

seq_len <- best_kwargs$seq_len
model <- NgramModel(vocab_size, seq_len, best_kwargs$smoothing)
for (tape in dataloader(train_tokens, seq_len)) {
  model$train(tape)
}

# ----------------------------
# Sampling
tape <- rep(EOT_TOKEN, seq_len - 1)
for (i in 1:200) {
  probs <- model$get_probs(tape)
  r <- random_float()
  sampled_index <- which(cumsum(probs) >= r)[1]
  next_token <- sampled_index - 1
  next_char <- token_to_char[[as.character(next_token)]]
  cat(next_char)
  tape <- c(tape[-1], next_token)
}
cat("\n")

# ----------------------------
# Final test evaluation
test_loss <- eval_split(model, test_tokens, seq_len)
test_perplexity <- exp(test_loss)
cat(sprintf("test_loss %f, test_perplexity %f\n", test_loss, test_perplexity))
