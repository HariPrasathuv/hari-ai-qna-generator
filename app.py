import os
os.environ["HF_HOME"] = "/tmp"



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from random import shuffle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

model_options = SentenceTransformer('all-MiniLM-L6-v2')


def gen(prompt,model,tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)

    # Decode and print the result
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("Generated", question)
    return output

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


@app.route('/generate', methods=['POST'])
def generate():
    file = request.files['file']
    if file:
        text = file.read().decode('utf-8')

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # ~1–2 sentences
            chunk_overlap=70  # helps retain context
        )
        chunks = splitter.split_text(text)

        # Load the model and tokenizer
        '''model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
'''

        # Load pre-trained embedding model
        
        candidate_words = list(set(text.split()))

        # Embed correct answer and candidates
        candidate_embeddings = model_options.encode(candidate_words, convert_to_tensor=True)

        
        mcqs = []
        for chunk in chunks:
            prompt = f"Generate question based on the following text:\n{chunk}"
            ques = gen(prompt,model,tokenizer)
            if ques[0] not in ['W','w','H','h','C','c','D','d']:
                continue
            prompt = f"Answer the following question : \n{ques} based on the following text:\n{chunk}"
            Answer = gen(prompt,model,tokenizer)

            if len(Answer.split())<2:

                # Correct answer
                correct_answer = (Answer.split())[-1]
                # Candidate pool (you can expand this list or load from a dictionary/thesaurus)

                correct_embedding = model_options.encode(correct_answer, convert_to_tensor=True)

                # Compute cosine similarity
                cos_scores = util.cos_sim(correct_embedding, candidate_embeddings)[0]
                # Sort and select top N most similar
                top_results = (sorted(zip(candidate_words, cos_scores), key=lambda x: x[1], reverse=True))

                options = [Answer]
                opt = [Answer[:4].lower()]


                for option in top_results:
                    if len(options)>=4:
                        break
                    if ((option[0][:4]).lower() not in opt) and option[0][0].isalpha() and option[0]!=Answer:
                        opt.append((option[0][:4]).lower())
                        options.append(option[0])
                if len(options)==4:
                    shuffle(options)
                    result_dict = {}
                    result_dict["Question"] = ques
                    result_dict["Options"] = options
                    result_dict["Answer"] = Answer
                    mcqs.append(result_dict)


        
        return jsonify(mcqs)
    return jsonify({"error": "No file uploaded"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
