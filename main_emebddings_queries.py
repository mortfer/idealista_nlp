import pandas as pd
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np

model_name = "intfloat/multilingual-e5-large"
model = TextEmbedding(model_name, threads=4)


# Encode the listings from a CSV
def get_listings(csv_file):
    # Read the CSV file containing the listings
    df = pd.read_csv(csv_file, dtype={"ASSETID": str})

    df = df[df["model_name"] == model_name]

    # Convert embeddings to a list of NumPy arrays
    embeddings = (
        df["embedding"]
        .apply(lambda x: np.array([np.float32(i) for i in x.split(",")]))
        .tolist()
    )
    df = df.drop(columns=["embedding", "model_name"])

    text = pd.read_csv("data/comments-cleaned.csv", dtype={"ASSETID": str})
    text = text[~text["ASSETID"].duplicated()]
    df = df.set_index("ASSETID").join(text.set_index("ASSETID"), how="inner")
    return df, embeddings


# Step 2: Encode the input question
def encode_question(question):
    # Encode the input question using the FastEmbed model
    question_embedding = model.embed([question])
    return list(question_embedding)


# Step 3: Find most similar listings
def find_similar_listings(
    question_embedding, listings_embeddings, listings_df, top_n=10
):
    # Calculate cosine similarity between the question and all listings
    similarities = cosine_similarity(question_embedding, listings_embeddings)
    # Add similarity scores to DataFrame
    listings_df["similarity"] = similarities.squeeze()
    # Return the most similar listings
    return listings_df.sort_values(by="similarity", ascending=False).head(top_n)


# Main script
def main():
    time1 = time.time()
    # Step 1: Encode the listings from a CSV
    listings_df, listings_embeddings = get_listings("embeddings.csv")

    # Step 2: Input a question and encode it
    # question = input("Enter your question: ")
    # question = " Busco piso con 3 habitaciones y piscina"
    # question = " Busco piso con terraza"
    question = " Busco piso con terraza"
    question_embedding = encode_question(question)

    # Step 3: Find and print the most similar listings
    similar_listings = find_similar_listings(
        question_embedding, listings_embeddings, listings_df
    )
    similar_listings.to_csv("most_similar.csv", index=True)
    print("\nMost similar listings to your question:")
    print(similar_listings)
    print("time:", round(time.time() - time1, 2))


if __name__ == "__main__":
    main()
