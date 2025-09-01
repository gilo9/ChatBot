# Premier Football AI Chatbot System

## Overview

This project is an intelligent chatbot system focused on football (soccer), designed for the Artificial Intelligence module (ISYS30221) at Nottingham Trent University. The chatbot leverages a range of AI techniques to answer questions, reason logically, and recognize images related to major football leagues, clubs, and players.

---

## Table of Contents

- [Features](#features)
- [Skills & Technologies](#skills--technologies)
- [AI Concepts](#ai-concepts)
- [System Architecture](#system-architecture)
- [Sample Workflows](#sample-workflows)
- [Learning Outcomes](#learning-outcomes)
- [Extra Functionalities](#extra-functionalities)
- [Documentation & Demo](#documentation--demo)

---

## Features

- **Football Q&A**: Answers questions about clubs, players, matches, and rules via pattern-based (AIML) and similarity-based (NLP) matching.
- **Real-time Football Data**: Integrates with football-data.org API for current Premier League standings, top scorers, match results, team squads, and coaches.
- **Logic-based Reasoning**: Supports knowledge base queries and dynamic logical fact addition using first-order logic (NLTK).
- **Image Recognition**: Classifies uploaded football league logos via a trained CNN model.
- **Extensible Knowledge Base**: CSV-based Q&A pairs and logic rules for easy updates and expansion.
- **Hybrid Multi-Agent System**: Combines rule-based, statistical, and neural approaches for robust conversational AI.

---

## Skills & Technologies

- **Python** (core language)
- **AIML** (Artificial Intelligence Markup Language)
- **TensorFlow/Keras** (deep learning, CNN for image classification)
- **NLTK** (Natural Language Processing, logic inference)
- **Pandas, Scikit-learn** (data handling, TF-IDF, cosine similarity)
- **http.client, httpx** (API integration)
- **Arrow** (date/time handling)
- **Wikipedia API** (external knowledge lookup)
- **Modular software engineering** (separation of concerns, extensibility)

---

## AI Concepts

- **Pattern-based Dialogue**: AIML rules for direct conversational matching.
- **Similarity-based Q&A**: TF-IDF vectorization and cosine similarity for flexible question answering.
- **First-order Logic Reasoning**: Dynamic fact checking, contradiction detection, and inference.
- **Deep Learning (CNN)**: Image classification of football league logos.
- **Hybrid Architecture**: Rule-based, statistical, and neural models working together.

---

## System Architecture

- `mybot-basic.py`: Main orchestrator, manages agents and conversation flows.
- `mybot-basic.xml`: AIML file for conversational rules.
- `QAPairs.csv`: Extensive Q&A pairs for similarity-based answers.
- `logic-kb.csv`: Logical knowledge base for reasoning.
- `APIFootball.py`: Handles football-data.org API interactions.
- `top3Leageue_classifier.h5`: Pre-trained CNN model for league logo recognition.

---

## Sample Workflows

1. **Manager Lookup**  
   *User*: "Who is the manager of Manchester United?"  
   *Bot*: Uses AIML/API to fetch and reply with current manager.

2. **Flexible Q&A**  
   *User*: Asks any football-related question not matched in AIML.  
   *Bot*: Applies NLP preprocessing, finds best CSV Q&A match via similarity.

3. **Logic Fact Addition**  
   *User*: "I know that Tim is British"  
   *Bot*: Checks for contradictions, adds fact to logic KB, confirms learning.

4. **Logic Query**  
   *User*: "Check that Tim is European"  
   *Bot*: Uses logic inference to verify fact, responds with "Correct", "Incorrect", or "I don't know".

5. **Image Classification**  
   *User*: Uploads a league logo image.  
   *Bot*: Uses CNN to classify and returns predicted league.

---

## Learning Outcomes

- **AI Integration**: Demonstrated ability to combine conversational AI, NLP, logic reasoning, and deep learning.
- **Practical NLP**: Experience with tokenization, lemmatization, stopwords, TF-IDF, and cosine similarity.
- **Logic & Knowledge Representation**: Built a dynamic, extensible knowledge base with logical inference.
- **API and Data Integration**: Real-time consumption and presentation of football data.
- **Software Engineering**: Modular, extensible codebase with clear separation of components.
- **Professional Portfolio**: Evidence of skills relevant for AI industry and academia.

---

## Extra Functionalities

- API integration for live football data.
- Dynamic knowledge base with contradiction checking and logic inference.
- CNN-based image classification (extensible to multi-object detection).
- Modular design for future expansion (e.g., multi-lingual, voice, fuzzy logic).

---

## Documentation & Demo

A full documentation is provided within the repository.
---

## License

This project is submitted as part of university coursework and is intended for educational purposes.

---

## Author

Developed by [gilo9](https://github.com/gilo9) for ISYS30221 Artificial Intelligence coursework, Nottingham Trent University.
