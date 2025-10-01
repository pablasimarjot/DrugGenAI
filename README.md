# DrugGenAI

This tool is an end-to-end framework for generative molecular design and analysis using SMILES strings as the molecular representation. It integrates deep learning (Keras/TensorFlow) with chemoinformatics (RDKit via Python/reticulate) to both generate novel molecules and evaluate their physicochemical properties.

**1. Data Preprocessing**

Input: SMILES strings (Simplified Molecular Input Line Entry System).

Steps:

Characters in the SMILES vocabulary are mapped to integers (char_to_int).

Each SMILES string is tokenized, padded to a uniform maximum length (max_length), and converted into integer sequences.

Output is a dataset suitable for feeding into a neural sequence model.

This ensures that variable-length SMILES strings are standardized for Transformer input.

**2. Transformer Model for Molecular Generation
**
The core of the tool is a custom Transformer-based neural network implemented in Keras/TensorFlow:

Input Layer: Encodes tokenized SMILES strings.

Embedding Layer: Converts tokens into dense embeddings (embedding_dim).

Positional Encoding: Provides order information to the model.

Transformer Blocks: Multiple layers (num_transformer_blocks) of multi-head self-attention (num_heads) and feed-forward networks (ff_dim).

Output Layer: Predicts the probability distribution of the next character in the SMILES vocabulary (softmax activation).

The model is trained in an auto-regressive manner, learning to reconstruct SMILES strings and eventually generate new valid molecules.

**3. Molecule Generation**

Starting from a user-provided seed string (e.g., "C"), the trained model:

Iteratively predicts the next character.

Samples from the probability distribution with temperature scaling (controls creativity).

Concatenates predictions until the maximum length is reached or an end token is encountered.

The result is a novel SMILES string, potentially representing a new chemical structure.

**4. Integration with RDKit for Property Calculation**

Through reticulate, the tool bridges R and Python to access RDKit:

Converts generated SMILES to molecular objects.

Computes key physicochemical properties, including:

Molecular Weight

LogP (lipophilicity)

H-bond Donors / Acceptors

Polar Surface Area (PSA)

Rotatable Bonds

Validates SMILES syntax and structure.

This ensures that generated molecules are chemically meaningful and provides descriptors for further drug-likeness analysis.

**5. High-Level Pipeline: Generate & Analyze**

The function generate_and_analyze_smiles() integrates generation and analysis:

Generate n new molecules from a seed.

Validate and compute RDKit descriptors.

Return a structured list of molecules with both SMILES strings and property profiles.

Results can then be summarized in human-readable format using print_molecule_properties().

**6. Applications**

This tool can be applied in:

De novo drug discovery: Generating novel candidate molecules.

Virtual screening: Producing chemical libraries enriched with drug-like properties.

Chemoinformatics research: Exploring chemical space using deep learning.

AI-driven molecule design: Integrating generative models with structure-based filtering.

**7. Customizability
**
Adjustable hyperparameters: embedding_dim, num_heads, num_transformer_blocks, batch_size, epochs.

Flexible generation: Different seeds, temperatures, and numbers of molecules.

Extensible property calculation: Additional RDKit descriptors can be integrated.

**In summary:**
This tool is a Transformer-based generative chemistry platform that not only creates new molecular candidates as SMILES strings but also evaluates their drug-relevant physicochemical properties. It bridges deep learning (for creativity) with RDKit chemoinformatics (for validity and property profiling), making it a practical asset in AI-driven drug discovery workflows.
