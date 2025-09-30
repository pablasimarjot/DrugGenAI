# Load required libraries
library(keras)
library(tensorflow)
library(stringr)

# Define parameters
max_length <- 100  # Maximum length of SMILES string
vocab_size <- 100  # Size of vocabulary (adjust based on your dataset)
embedding_dim <- 256
num_heads <- 8
ff_dim <- 512
num_transformer_blocks <- 4
batch_size <- 32
epochs <- 50

# Function to preprocess SMILES strings
preprocess_smiles <- function(smiles, max_length) {
  # Create a character to integer mapping
  chars <- unique(unlist(strsplit(smiles, "")))
  char_to_int <- seq_along(chars)
  names(char_to_int) <- chars
  
  # Pad sequences
  smiles_padded <- lapply(strsplit(smiles, ""), function(x) {
    c(x, rep("", max_length - length(x)))
  })
  
  # Convert to integer sequences
  smiles_int <- lapply(smiles_padded, function(x) {
    sapply(x, function(char) ifelse(char == "", 0, char_to_int[char]))
  })
  
  return(list(smiles_int = smiles_int, char_to_int = char_to_int))
}

# Load and preprocess your SMILES data
# smiles_data <- read.csv("your_smiles_data.csv")
# smiles <- smiles_data$SMILES
# processed_data <- preprocess_smiles(smiles, max_length)

# Create the Transformer model
create_transformer_model <- function(max_length, vocab_size, embedding_dim, num_heads, ff_dim, num_transformer_blocks) {
  inputs <- layer_input(shape = c(max_length))
  
  x <- inputs %>%
    layer_embedding(input_dim = vocab_size, output_dim = embedding_dim) %>%
    layer_position_embedding(input_dim = max_length, output_dim = embedding_dim)
  
  for (i in 1:num_transformer_blocks) {
    x <- x %>%
      layer_transformer(num_heads = num_heads, ff_dim = ff_dim)
  }
  
  outputs <- x %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(units = vocab_size, activation = "softmax")
  
  model <- keras_model(inputs = inputs, outputs = outputs)
  
  model %>% compile(
    optimizer = optimizer_adam(),
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  return(model)
}

# Create and train the model
# model <- create_transformer_model(max_length, vocab_size, embedding_dim, num_heads, ff_dim, num_transformer_blocks)
# 
# history <- model %>% fit(
#   x = processed_data$smiles_int,
#   y = processed_data$smiles_int,
#   batch_size = batch_size,
#   epochs = epochs,
#   validation_split = 0.2
# )

# Function to generate new SMILES strings
generate_smiles <- function(model, start_string, max_length, char_to_int, temperature = 1.0) {
  int_to_char <- names(char_to_int)
  input_eval <- sapply(strsplit(start_string, "")[[1]], function(char) char_to_int[char])
  input_eval <- array(c(input_eval, rep(0, max_length - length(input_eval))), dim = c(1, max_length))
  
  generated_smiles <- start_string
  
  for (i in 1:(max_length - nchar(start_string))) {
    predictions <- predict(model, input_eval)
    predictions <- predictions[1, nchar(start_string) + i, ] / temperature
    predicted_id <- sample(seq_along(predictions), size = 1, prob = softmax(predictions))
    
    input_eval[1, nchar(start_string) + i] <- predicted_id
    generated_smiles <- paste0(generated_smiles, int_to_char[predicted_id])
    
    if (int_to_char[predicted_id] == "") {
      break
    }
  }
  
  return(generated_smiles)
}

# Generate new SMILES strings
# new_smiles <- generate_smiles(model, "C", max_length, processed_data$char_to_int)
# print(new_smiles)


# Load required libraries
library(keras)
library(tensorflow)
library(stringr)
library(reticulate)

# Initialize Python and import RDKit
use_python("/path/to/your/python") # Adjust this path to your Python installation
py_run_string("
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
")

# [Previous code remains the same up to the generate_smiles function]

# Function to calculate physicochemical properties
calculate_properties <- function(smiles) {
  mol <- py$Chem$MolFromSmiles(smiles)
  if (is.null(mol)) {
    return(list(
      molecular_weight = NA,
      logp = NA,
      num_hdonors = NA,
      num_hacceptors = NA,
      polar_surface_area = NA,
      rotatable_bonds = NA,
      valid = FALSE
    ))
  }
  
  properties <- list(
    molecular_weight = py$Descriptors$ExactMolWt(mol),
    logp = py$Crippen$MolLogP(mol),
    num_hdonors = py$Descriptors$NumHDonors(mol),
    num_hacceptors = py$Descriptors$NumHAcceptors(mol),
    polar_surface_area = py$Descriptors$TPSA(mol),
    rotatable_bonds = py$Descriptors$NumRotatableBonds(mol),
    valid = TRUE
  )
  
  return(properties)
}

# Function to generate new SMILES strings and calculate their properties
generate_and_analyze_smiles <- function(model, start_string, max_length, char_to_int, temperature = 1.0, num_molecules = 10) {
  results <- list()
  
  for (i in 1:num_molecules) {
    new_smiles <- generate_smiles(model, start_string, max_length, char_to_int, temperature)
    properties <- calculate_properties(new_smiles)
    
    results[[i]] <- c(list(smiles = new_smiles), properties)
  }
  
  return(results)
}

# Generate new SMILES strings and analyze their properties
# analyzed_molecules <- generate_and_analyze_smiles(model, "C", max_length, processed_data$char_to_int, num_molecules = 10)

# Function to print results in a readable format
print_molecule_properties <- function(molecules) {
  for (i in seq_along(molecules)) {
    cat("Molecule", i, "\n")
    cat("SMILES:", molecules[[i]]$smiles, "\n")
    if (molecules[[i]]$valid) {
      cat("Molecular Weight:", round(molecules[[i]]$molecular_weight, 2), "\n")
      cat("LogP:", round(molecules[[i]]$logp, 2), "\n")
      cat("H-Bond Donors:", molecules[[i]]$num_hdonors, "\n")
      cat("H-Bond Acceptors:", molecules[[i]]$num_hacceptors, "\n")
      cat("Polar Surface Area:", round(molecules[[i]]$polar_surface_area, 2), "\n")
      cat("Rotatable Bonds:", molecules[[i]]$rotatable_bonds, "\n")
    } else {
      cat("Invalid SMILES string\n")
    }
    cat("\n")
  }
}

# Print the properties of the generated molecules
# print_molecule_properties(analyzed_molecules)

# After training your model
analyzed_molecules <- generate_and_analyze_smiles(model, "C", max_length, processed_data$char_to_int, num_molecules = 10)
print_molecule_properties(analyzed_molecules)