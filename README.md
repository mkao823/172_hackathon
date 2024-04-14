# Food Safety AI Response System

This project implements an AI response system for answering food safety-related questions. It uses OpenAI's GPT-3 technology to generate responses based on input queries and a Chroma database to store and retrieve relevant information.

## Features

- **AI-Powered Responses**: Leverages GPT-3 to provide accurate and contextually relevant answers.
- **Document Embedding**: Uses Chroma for efficient document similarity search.
- **User-Friendly Interface**: Includes a command-line interface for easy interaction with the system.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

### Installing 
1. Clone the repository

git clone https://yourrepositorylink.com
cd your-project-folder

2. Install required packages

pip install -r requirements.txt

3. Set environmnent variables

export OPENAI_API_KEY='your_openai_api_key_here'

### Usage
Run the application using the following command:
python query_data.py "Enter your question here"
