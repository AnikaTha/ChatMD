# ChatMD

LoRA fine-tuned Tiny-Llama-1.1B model for medical question-answering trained on the MedQA dataset.

# Installation

Recommended to create a virtual environment. Install dependencies using ```./requirements.txt```.

Model weights can be downloaded [here](https://drive.google.com/file/d/1JYjMXnhzBceJ3MpjVXyEnyBayQVjHIoM/view?usp=sharing)

## Datasets - ```./Data```

- ```/MIMICIII```
  - Structure:
    - context
      - “admission date: [2124-7-21] discharge date: [2124-8-18] service: medicine allergies: amlodipine…”
    - question
      - “does the patient have a current copd exacerbation”
    - answer
      - answer_start: “141” 
  - ```test.final.json```
     - contains q/a pairs for MIMIC
- ```/MedQA/questions/US```
  - Structure: 
    - context & question: "question": "A 36-year-old man is brought to the emergency department 40 minutes after being involved in a shooting. He sustained a gunshot wound in an altercation outside of a bar. On arrival, he is oriented ……. Which of the following is the most appropriate next step in management?", 
    - options: "options": {"A": "Chest tube insertion in the fifth intercostal space at the midaxillary line", "B": "CT scan of the chest, abdomen, and pelvis", "C": "Local wound exploration", "D": "Exploratory laparotomy", "E": "Video-assisted thoracoscopic surgery"}
    - answer: "answer": "Exploratory laparotomy", "answer_idx": "D"
  - ```train.json```
    - contains q/a pairs for mimic in json format
  - ```test.json```
    - contains q/a pairs for mimic in json format


## Code structure

- ```./notebooks/tiny-llama.ipynb```
  - Primary notebook for LoRA fine-tuning of TinyLlama. Loads training data and tokenizes using a preprocessing function. Sets up a LoRA configuration using PEFT to train.
- ```./notebooks/tiny-llama.py```
  - tiny-llama.ipynb converted into a python script to train using batch scripts.
- ```./notebooks/baseline.ipynb```
  - Uses pipeline objects from Hugging Face to generate preliminary responses to questions.
- ```./notebooks/eval.ipynb```
  - Inference notebook to generate responses using the fine-tuned model.
- ```./train.sh```
  - Training script to run on GPU cluster
- ```./run.sh```
  - Batch script we used to run on VT ARC





## Baseline

TinyLlama1B - Tested on the 1 Billion parameter model to set a baseline with the MedQA dataset. Obtained a 37.5% accuracy on the sample Q/A pairs.

MedAlpaca7B - Tested the 7 Billion parameter model to set a baseline with the MedQA dataset. Obtained a 40.625% accuracy on the sample Q/A pairs. 

Train set:
```./Data/MedQA/data_clean/questions/US/train.jsonl``` - The training data for the MedQA dataset.

Test set:
```./Data/MedQA/data_clean/questions/US/test.jsonl``` - The test data for the MedQA dataset. 

## Finetuning - 

- Model: TinyLlama-1.1B
- Fine-tuning Strategy: LoRA
- Batch Size: 8 per device
- Learning Rate: 1e-5
- Loss Function: Hugging Face CLM Default



Finetuned tiny-llama 1.1B model available here:


## References

[1] Monica Agrawal, Stefan Hegselmann, Hunter Lang, Yoon Kim, and
David Sontag. 2022. Large language models are few-shot clinical infor-
mation extractors. In Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing. 1998–2022. \n

[2] Scott L. Fleming, Alejandro Lozano, William J. Haberkorn, Jenelle A.
Jindal, Eduardo P. Reis, Rahul Thapa, Louis Blankemeier, Julian Z. Genk-
ins, Ethan Steinberg, Ashwin Nayak, Birju S. Patel, Chia-Chun Chiang,
Alison Callahan, Zepeng Huo, Sergios Gatidis, Scott J. Adams, Oluseyi
Fayanju, Shreya J. Shah, Thomas Savage, Ethan Goh, Akshay S. Chaud-
hari, Nima Aghaeepour, Christopher Sharp, Michael A. Pfeffer, Percy
Liang, Jonathan H. Chen, Keith E. Morse, Emma P. Brunskill, Jason A.
Fries, and Nigam H. Shah. 2023. MedAlign: A Clinician-Generated
Dataset for Instruction Following with Electronic Medical Records.
arXiv:2308.14089 [cs.CL] \n

[3] Tianyu Han, Lisa C. Adams, Jens-Michalis Papaioannou, Paul Grund-
mann, Tom Oberhauser, Alexander Löser, Daniel Truhn, and Keno K.
Bressem. 2023. MedAlpaca – An Open-Source Collection of
Medical Conversational AI Models and Training Data. (2023).
arXiv:2304.08247 [cs.CL] \n

[4] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang,
and Peter Szolovits. 2020. What Disease does this Patient Have? A
Large-scale Open Domain Question Answering Dataset from Medical
Exams. CoRR abs/2009.13081 (2020). arXiv:2009.13081 https://arxiv.org/
abs/2009.13081\n

[5] Alistair E.W. Johnson, Tom J. Pollard, Lu Shen, Li-wei H. Lehman,
Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter
Szolovits, Leo Anthony Celi, and Roger G. Mark. 2016. MIMIC-III,
a freely accessible critical care database. Scientific Data 3, 1 (May 2016).
https://doi.org/10.1038/sdata.2016.35
[6] Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Marzyeh
Ghassemi, Michael C. Hughes, and Tristan Naumann. 2020. MIMIC-
Extract: a data extraction, preprocessing, and representation pipeline for
MIMIC-III. (2020), 222–235. https://doi.org/10.1145/3368555.3384469



