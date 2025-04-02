import pandas as pd
import fastembed
import os
import time

time0 = time.time()
# Define paths
csv_path = "data/comments-cleaned.csv"  # Change this to your CSV file path
output_path = "embeddings.csv"  # Change this to where you want to save the embeddings

# Load data
df = pd.read_csv(csv_path)
df = df[~df[["ASSETID"]].duplicated()]

# Initialize FastEmbed model
model_name = "intfloat/multilingual-e5-large"  # Store model name for reference
model = fastembed.TextEmbedding(model_name, threads=6)


# Generate embeddings
texts = df["TEXT"].tolist()
embeddings = list(model.embed(texts))  # Adjust batch size if needed

# Convert embeddings to a string format (or use JSON)
df["embedding"] = [",".join(map(str, emb)) for emb in embeddings]
df["model_name"] = model_name

# Drop the 'text' column
df = df.drop(columns=["TEXT"])

# Check if the output file already exists
if os.path.exists(output_path):
    # If the file exists, load the existing DataFrame
    existing_df = pd.read_csv(output_path)

    # Remove rows with the same assetid and model_name from the existing DataFrame
    new_df = df[~df[["ASSETID", "model_name"]].duplicated()]

    # Make sure we're not duplicating assetid and model_name entries
    existing_keys = existing_df[["ASSETID", "model_name"]]
    new_df = new_df[~new_df[["ASSETID", "model_name"]].isin(existing_keys).all(axis=1)]

    # Append new data to the existing DataFrame
    if not new_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"New embeddings appended to {output_path}")
    else:
        print("No new asset IDs to add.")
else:
    # If the file doesn't exist, simply save the DataFrame
    df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path}")

print("time:", round(time.time() - time0, 2))
