#https://blogs.rstudio.com/tensorflow/posts/2017-12-22-word-embeddings-with-keras/

current_wd <-  getwd()
current_wd

current_wd <- "C:/Users/Brandon_YEO/Desktop"
setwd(current_wd)

#download.file("https://snap.stanford.edu/data/finefoods.txt.gz","finefoods.txt.gz")
####################################################################################
####################################################################################
####################################################################################
####################################################################################

##https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa

library(dplyr)
library(readr)
library(stringr)

reviews <- read_lines("finefoods.txt.gz")
reviews <- reviews[str_sub(reviews, 1, 12) == "review/text:"]
reviews <- str_sub(reviews, start = 14)
reviews <- iconv(reviews, to = "UTF-8")

head(reviews, 2)

unique_words  <- length(unique(unlist(str_split(reviews[1:200], " "))))
unique_words

####################################################################################

library(tensorflow)
#install_tensorflow(version = "gpu")

library(keras)
#install_keras(tensorflow = "gpu")

tokenizer <- text_tokenizer(num_words = 20000)
tokenizer %>% fit_text_tokenizer(reviews)

library(reticulate)
library(purrr)

#https://becominghuman.ai/how-does-word2vecs-skip-gram-work-f92e0525def4

skipgrams_generator <-
  function(text,
           tokenizer,
           window_size,
           negative_samples) {
    gen <- texts_to_sequences_generator(tokenizer, sample(text))
    function() {
      skip <- generator_next(gen) %>%
        skipgrams(
          vocabulary_size = tokenizer$num_words,
          window_size = window_size,
          negative_samples = 1
        )
      x <-
        transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
      y <- skip$labels %>% as.matrix(ncol = 1)
      list(x, y)
    }
  }

####################################################################################

embedding_size <- 128  # Dimension of the embedding vector.
skip_window <- 5       # How many words to consider left and right.
num_sampled <-
  1       # Number of negative examples to sample for each word.

input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)

embedding <- layer_embedding(
  input_dim = tokenizer$num_words + 1,
  output_dim = embedding_size,
  input_length = 1,
  name = "embedding"
)

target_vector <- input_target %>%
  embedding() %>%
  layer_flatten()

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

dot_product <-
  layer_dot(list(target_vector, context_vector), axes = 1)

#dot_product
#https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/

output <-
  layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam", metrics='accuracy')

summary(model)


####################################################################################

history <- model %>%
  fit_generator(
    skipgrams_generator(reviews, tokenizer, skip_window, negative_samples),
    steps_per_epoch = 100000,
    epochs = 10
  )

plot(history, method = "base")



embedding_matrix <- get_weights(model)[[1]]

words <- data_frame(word = names(tokenizer$word_index),
                    id = as.integer(unlist(tokenizer$word_index)))

words <- words %>%
  filter(id <= tokenizer$num_words) %>%
  arrange(id)

row.names(embedding_matrix) <- c("UNK", words$word)

####################################################################################

library(text2vec)

find_similar_words <- function(word, embedding_matrix, n = 5) {
  similarities <- embedding_matrix[word, , drop = FALSE] %>%
    sim2(embedding_matrix, y = ., method = "cosine")
  
  similarities[, 1] %>% sort(decreasing = TRUE) %>% head(n)
}

find_similar_words("them",embedding_matrix,10)
####################################################################################

library(Rtsne)
library(ggplot2)
library(plotly)

tsne <-
  Rtsne(embedding_matrix[2:700, ], perplexity = 50, pca = TRUE)

tsne_plot <- tsne$Y %>%
  as.data.frame() %>%
  mutate(word = row.names(embedding_matrix)[2:700]) %>%
  ggplot(aes(x = V1, y = V2, label = word)) +
  geom_text(size = 1)
tsne_plot



