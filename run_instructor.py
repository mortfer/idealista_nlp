from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-large", device="cpu")
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction, sentence]])
print(embeddings)
