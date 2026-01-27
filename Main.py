import streamlit as st
import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from fpdf import FPDF
import re
import random
import time
import concurrent.futures
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import requests
from datetime import datetime
import tempfile
import json
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pandas as pd

EDUCATIONAL_BOARDS = ["CBSE", "ICSE", "State Board"]
CLASSES = list(range(1, 13))
SUBJECTS = ["Mathematics", "Science", "Social Studies", "English", "Hindi", "Physics", "Chemistry", "Biology", "Computer Science", "Other"]
QUESTION_TYPES = ["multiple-choice", "short-answer", "long-answer", "true-false", "fill-in-the-blanks", "match-the-following"]

def init_session_state():
    defaults = {
        'vectors': None,
        'pdf_ready': False,
        'answers_generated': False,
        'question_settings': None,
        'ocr_enabled': True,
        'raw_text': None,
        'ai_answer': None,
        'answers': None,
        'question_bank': [],
        'generation_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_api_key(key_name):
    if key_name in st.secrets:
        return st.secrets[key_name]
    elif key_name in os.environ:
        return os.environ[key_name]
    else:
        st.error(f"Missing API key: {key_name}. Please add it to your secrets.")
        return None

@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings model (cached for performance)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def estimate_tokens(text):
    # Simple estimation: ~1.3 tokens per word for English
    words = text.split()
    return int(len(words) * 1.3)

def validate_inputs(board, class_level, subject, num_questions, question_type):
    errors = []
    
    if not board or board not in EDUCATIONAL_BOARDS:
        errors.append("⚠️ Invalid educational board selected")
    
    if not (1 <= class_level <= 12):
        errors.append("⚠️ Class level must be between 1 and 12")
    
    if not subject or subject not in SUBJECTS:
        errors.append("⚠️ Invalid subject selected")
    
    if not (5 <= num_questions <= 25):
        errors.append("⚠️ Number of questions must be between 5 and 25")
    
    if not question_type or question_type not in QUESTION_TYPES:
        errors.append("⚠️ Invalid question type selected")
    
    return len(errors) == 0, errors

def is_pdf_scanned(pdf_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        first_page = pdf_reader.pages[0]
        text = first_page.extract_text()
        
        # If very little text is extracted, it's likely a scanned PDF
        if len(text.strip()) < 100:
            return True
        return False
    except Exception:
        return True

def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def process_scanned_pdf(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        
        with st.spinner("Performing OCR on scanned PDF..."):
            # Use ThreadPoolExecutor for parallel OCR processing
            with concurrent.futures.ThreadPoolExecutor() as executor:
                text_chunks = list(executor.map(extract_text_from_image, images))
        
        combined_text = "\n\n".join([chunk for chunk in text_chunks if chunk])
        return combined_text
    except Exception as e:
        st.error(f"Failed to process scanned PDF: {e}")
        return ""

def get_pdf_text(upload_pdf):
    if upload_pdf is None:
        st.error("No PDF file uploaded.")
        return None
    
    try:
        pdf_bytes = upload_pdf.getvalue()
        
        # Validate PDF size (max 50MB)
        if len(pdf_bytes) > 50 * 1024 * 1024:
            st.error("PDF file is too large. Please upload a file smaller than 50MB.")
            return None
        
        if is_pdf_scanned(pdf_bytes):
            st.info("Detected a scanned/image-based PDF. Using OCR to extract text...")
            return process_scanned_pdf(pdf_bytes)
        
        pdf_reader = PyPDF2.PdfReader(upload_pdf)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n\n"

        return extracted_text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return None

def split_data_into_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def store_chunks_vectorDB(chunks, embedding):
    batch_size = 100
    vector_db = None    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_num = i // batch_size
        progress_bar.progress((batch_num + 1) / total_batches)
        status_text.text(f"Processing batch {batch_num + 1} of {total_batches}...")
        
        try:
            if vector_db is None:
                vector_db = FAISS.from_texts(texts=batch_chunks, embedding=embedding)
            else:
                vector_db.add_texts(texts=batch_chunks)
            
            # Small delay to prevent any potential issues
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error processing batch {batch_num + 1}: {e}")
            raise
    
    progress_bar.empty()
    status_text.empty()
    return vector_db

def parse_questions_to_list(content):
    """Parse questions text into structured list for export"""
    lines = content.split('\n')
    questions_list = []
    
    current_question = ""
    current_number = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a question number
        if line[0].isdigit() and '.' in line[:3]:
            if current_question:
                questions_list.append({
                    'number': current_number,
                    'text': current_question.strip()
                })
            
            current_number += 1
            current_question = line
        else:
            current_question += " " + line
    
    # Add the last question
    if current_question:
        questions_list.append({
            'number': current_number,
            'text': current_question.strip()
        })
    
    return questions_list

def create_word_doc(content, board, class_level, subject, question_type, is_answer_key=False):
    """Create a Word document with the generated content"""
    doc = Document()
    
    # Add title
    title = doc.add_heading(f"{board} Class {class_level} - {subject}", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add subtitle
    subtitle_text = f"{question_type.title()} {'Answer Key' if is_answer_key else 'Questions'}"
    subtitle = doc.add_heading(subtitle_text, level=2)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add date
    date_para = doc.add_paragraph()
    date_run = date_para.add_run(f"Generated on: {datetime.now().strftime('%d-%m-%Y')}")
    date_run.italic = True
    date_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    doc.add_paragraph()
    
    # Add content
    for line in content.split('\n'):
        if line.strip():
            # Check if it's a question number or section header
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or \
               line.strip().startswith(('Question', 'Q.', 'I.', 'II.')):
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
                run.font.size = Pt(12)
            # Check if it's an option
            elif line.strip().startswith(('A)', 'B)', 'C)', 'D)', 'a)', 'b)', 'c)', 'd)')):
                p = doc.add_paragraph(line, style='List Bullet')
            else:
                doc.add_paragraph(line)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
    doc.save(temp_file.name)
    temp_file.close()
    
    return temp_file.name

def create_excel_file(content, board, class_level, subject, question_type, is_answer_key=False):
    """Create an Excel file with questions/answers in structured format"""
    questions_list = parse_questions_to_list(content)
    
    # Create DataFrame
    data = []
    for q in questions_list:
        data.append({
            'No.': q['number'],
            'Question/Answer': q['text'],
            'Type': question_type,
            'Category': 'Answer' if is_answer_key else 'Question'
        })
    
    df = pd.DataFrame(data)
    
    # Add metadata sheet
    metadata = {
        'Board': [board],
        'Class': [class_level],
        'Subject': [subject],
        'Question Type': [question_type],
        'Generated On': [datetime.now().strftime('%d-%m-%Y %H:%M:%S')],
        'Total Items': [len(data)]
    }
    df_metadata = pd.DataFrame(metadata)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    
    with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Questions', index=False)
        df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    temp_file.close()
    return temp_file.name

def create_json_export(content, settings):
    """Create JSON export with metadata"""
    questions_list = parse_questions_to_list(content)
    
    # Create JSON structure
    export_data = {
        'metadata': {
            'board': settings.get('board', 'Unknown'),
            'class': settings.get('class', 'Unknown'),
            'subject': settings.get('subject', 'Unknown'),
            'question_type': settings.get('type', 'Unknown'),
            'complexity': settings.get('complexity', 'Unknown'),
            'generated_at': datetime.now().isoformat(),
            'total_questions': len(questions_list)
        },
        'questions': questions_list
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def save_to_question_bank(questions, answers, settings):
    """Save generated questions to question bank"""
    if 'question_bank' not in st.session_state:
        st.session_state.question_bank = []
    
    entry = {
        'id': len(st.session_state.question_bank) + 1,
        'timestamp': datetime.now().isoformat(),
        'settings': settings,
        'questions': questions,
        'answers': answers,
        'tags': [settings['board'], settings['subject'], settings['type']]
    }
    
    st.session_state.question_bank.append(entry)
    
    # Also add to generation history
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    st.session_state.generation_history.append({
        'timestamp': datetime.now(),
        'settings': settings,
        'question_count': len(parse_questions_to_list(questions)),
        'preview': questions[:100] + "..."
    })

def create_diverse_question_prompt(question_type, board, class_level, subject, seed=None):
    if seed is None:
        seed = random.randint(1, 1000)
    
    approaches = [
        "analytical thinking",
        "critical reasoning",
        "real-world application",
        "conceptual understanding",
        "creative problem-solving"
    ]
    
    if subject == "Mathematics":
        focus_areas = random.sample([
            "algebraic concepts", "geometric principles", "statistical analysis",
            "numerical reasoning", "mathematical proofs", "applied mathematics"
        ], k=2)
    elif subject in ["Physics", "Chemistry", "Biology", "Science"]:
        focus_areas = random.sample([
            "scientific principles", "experimental design", "natural phenomena",
            "scientific theories", "laboratory techniques", "scientific discoveries"
        ], k=2)
    else:
        focus_areas = random.sample([
            "key concepts", "important theories", "significant events",
            "critical analysis", "practical applications", "foundational principles"
        ], k=2)
    
    cognitive_levels = random.sample([
        "knowledge recall", "comprehension", "application",
        "analysis", "synthesis", "evaluation"
    ], k=3)
    
    base_prompt = f"""
    You are an expert {board} Class {class_level} {subject} teacher.
    
    Generate ONLY {question_type} questions based on the context provided.
    
    To ensure DIVERSE and HIGH-QUALITY questions:
    - Focus on {focus_areas[0]} and {focus_areas[1]}
    - Target cognitive levels of {cognitive_levels[0]}, {cognitive_levels[1]}, and {cognitive_levels[2]}
    - Use an approach emphasizing {random.choice(approaches)}
    - Ensure questions are varied in difficulty and style
    - Use creativity seed #{seed} to make questions unique and non-repetitive
    
    IMPORTANT: 
    1. Create ONLY {question_type} questions - do not mix question types
    2. Do not include any explanations, notes, or additional text
    3. Start directly with "1." - no preamble
    4. Number your questions sequentially
    
    Context: {{context}}
    Complexity Level: {{complexity_level}}
    Number of Questions: {{num_questions}}
    """
    
    if question_type == "multiple-choice":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Question text]?
           A) [Option A]
           B) [Option B]
           C) [Option C]
           D) [Option D]
        
        2. [Next question]?
           A) [Option A]
           B) [Option B]
           C) [Option C]
           D) [Option D]
        
        and so on.
        """
    
    elif question_type == "short-answer":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Question text]? (Answer in 30-50 words)
        
        2. [Next question]? (Answer in 30-50 words)
        
        and so on.
        """
    
    elif question_type == "long-answer":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Question text]? (Answer in 100-150 words)
        
        2. [Next question]? (Answer in 100-150 words)
        
        and so on.
        """
    
    elif question_type == "true-false":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Statement]. (True/False)
        
        2. [Statement]. (True/False)
        
        and so on.
        """
    
    elif question_type == "fill-in-the-blanks":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Sentence with __________ as blank]
        
        2. [Sentence with __________ as blank]
        
        and so on.
        """
    
    elif question_type == "match-the-following":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. Match the following:
           Column A              Column B
           a) [Item 1]          i) [Match A]
           b) [Item 2]          ii) [Match B]
           c) [Item 3]          iii) [Match C]
           d) [Item 4]          iv) [Match D]
        
        2. Match the following:
           Column A              Column B
           a) [Item 1]          i) [Match A]
           b) [Item 2]          ii) [Match B]
           c) [Item 3]          iii) [Match C]
           d) [Item 4]          iv) [Match D]
        
        and so on.
        """
    
    return base_prompt

def get_answer_prompt(question_type, board, class_level, subject):
    base_prompt = f"""
    You are an expert {board} Class {class_level} {subject} teacher.
    
    Below are {question_type} questions. Provide correct answers for each question.
    
    Questions:
    {{questions}}
    
    IMPORTANT:
    1. Provide answers in same numbered order as questions
    2. Start directly with "1." - no preamble
    3. Be clear, accurate, and educational in your answers
    """
    
    if question_type == "multiple-choice":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Correct Option Letter]: [Brief explanation why this is correct]
        
        2. [Correct Option Letter]: [Brief explanation why this is correct]
        
        and so on.
        """
    
    elif question_type == "short-answer" or question_type == "long-answer":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Complete answer that directly addresses the question]
        
        2. [Complete answer that directly addresses the question]
        
        and so on.
        """
    
    elif question_type == "true-false":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [TRUE or FALSE]: [Brief explanation supporting this answer]
        
        2. [TRUE or FALSE]: [Brief explanation supporting this answer]
        
        and so on.
        """
    
    elif question_type == "fill-in-the-blanks":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. [Word or phrase that fills the blank]: [Brief explanation if needed]
        
        2. [Word or phrase that fills the blank]: [Brief explanation if needed]
        
        and so on.
        """
    
    elif question_type == "match-the-following":
        base_prompt += """
        FORMAT STRICTLY AS:
        1. a-[correct number], b-[correct number], c-[correct number], d-[correct number]: [Brief explanation of relationships]
        
        2. a-[correct number], b-[correct number], c-[correct number], d-[correct number]: [Brief explanation of relationships]
        
        and so on.
        """
    
    return base_prompt

def check_question_format(text, question_type):
    lower_text = text.lower()
    
    if question_type == "multiple-choice" and ("a)" in lower_text or "a)" in lower_text or "a." in lower_text):
        return True
    elif question_type == "short-answer" and ("words" in lower_text or "answer" in lower_text):
        return True
    elif question_type == "long-answer" and ("words" in lower_text or "100" in lower_text):
        return True
    elif question_type == "true-false" and ("true/false" in lower_text or "true or false" in lower_text):
        return True
    elif question_type == "fill-in-the-blanks" and ("____" in lower_text or "..." in lower_text or "_____" in lower_text):
        return True
    elif question_type == "match-the-following" and "column" in lower_text:
        return True
    else:
        return False

def generate_questions(inputs, max_retries=2):
    content = inputs['content']
    question_type = inputs['question_types']
    complexity_level = inputs['complexity_level']
    num_questions = inputs['num_questions']
    board = inputs.get('board', 'CBSE')
    class_level = inputs.get('class_level', '10')
    subject = inputs.get('subject', 'General')
    
    # Validate inputs
    is_valid, errors = validate_inputs(board, class_level, subject, num_questions, question_type)
    if not is_valid:
        for error in errors:
            st.error(error)
        return "Please correct the errors above and try again."
    
    st.info(f"""
    Generating {num_questions} diverse {question_type} questions 
    for {board} Class {class_level} {subject} at {complexity_level} complexity
    """)
    
    st.session_state.question_settings = {
        "type": question_type,
        "board": board,
        "class": class_level,
        "subject": subject,
        "complexity": complexity_level,
        "count": num_questions
    }
    
    groq_api_key = get_api_key("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key is missing. Please check your secrets.")
        return "Error: Missing API key. Contact the administrator."
    
    llm = ChatGroq(api_key=groq_api_key, model='llama3-8b-8192')
    
    seed = random.randint(1, 1000)
    
    prompt_template = create_diverse_question_prompt(
        question_type, 
        board, 
        class_level, 
        subject,
        seed=seed
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    if st.session_state.vectors:
        search_query = f"{board} Class {class_level} {subject} {question_type} questions {complexity_level} complexity"
        
        with st.spinner('Retrieving relevant content from document...'):
            k_chunks = 5
            relevant_docs = st.session_state.vectors.similarity_search(search_query, k=k_chunks)
            relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            estimated_tokens = estimate_tokens(relevant_content)
            
            while estimated_tokens > 2000 and k_chunks > 1:
                k_chunks -= 1
                relevant_docs = st.session_state.vectors.similarity_search(search_query, k=k_chunks)
                relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])
                estimated_tokens = estimate_tokens(relevant_content)
                
            st.session_state.content_tokens = estimated_tokens
            
            if estimated_tokens > 2000:
                words = relevant_content.split()
                relevant_content = " ".join(words[:1500])
                
            content = relevant_content
    else:
        words = content.split()
        if len(words) > 1500:
            content = " ".join(words[:1500])
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                st.info(f"Retry attempt {attempt} with a different approach...")
            
            with st.spinner(f'AI is generating {question_type} questions (attempt {attempt+1})...'):
                formatted_prompt = prompt.format(
                    context=content,
                    complexity_level=complexity_level,
                    num_questions=num_questions
                )
                
                prompt_tokens = estimate_tokens(formatted_prompt)
                if prompt_tokens > 5000:
                    st.warning(f"Prompt too large (~{prompt_tokens} tokens). Reducing content size...")
                    words = content.split()
                    reduced_words = int(len(words) * (4000 / prompt_tokens))
                    content = " ".join(words[:reduced_words])
                    
                    formatted_prompt = prompt.format(
                        context=content,
                        complexity_level=complexity_level,
                        num_questions=num_questions
                    )
                
                start_time = time.time()
                response = llm.invoke(formatted_prompt)
                generation_time = time.time() - start_time
                
                st.session_state.last_generation_time = generation_time
            
            ai_answer = response.content if hasattr(response, 'content') else str(response)
            cleaned_answer = post_process_output(ai_answer)
            
            if check_question_format(cleaned_answer, question_type):
                st.success(f"Successfully generated {question_type} questions in {generation_time:.1f} seconds!")
                
                st.session_state.generated_questions = cleaned_answer
                return cleaned_answer
            
            if attempt == max_retries:
                st.warning(f"Output format needs adjustment. Reformatting...")
                fix_prompt = f"""
                I need {num_questions} {question_type} questions for {board} Class {class_level} {subject}.
                
                The content I have doesn't match the {question_type} format.
                
                Please reformat the following content into proper {question_type} questions:
                
                {cleaned_answer}
                
                Make sure to follow the proper format for {question_type} questions as described earlier.
                """
                
                with st.spinner("Reformatting questions..."):
                    fix_response = llm.invoke(fix_prompt)
                    cleaned_answer = post_process_output(fix_response.content)
                
                st.session_state.generated_questions = cleaned_answer
                return cleaned_answer
            
            seed = random.randint(1, 1000)
            prompt_template = create_diverse_question_prompt(question_type, board, class_level, subject, seed=seed)
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
        except Exception as e:
            st.error(f"Error during generation (attempt {attempt+1}): {e}")
            
            error_str = str(e)
            if "413" in error_str or "token" in error_str.lower() or "too large" in error_str.lower():
                words = content.split()
                content = " ".join(words[:len(words)//2])
                st.warning("Request was too large. Reducing content size and trying again...")
            
            if attempt == max_retries:
                fallback_content = subject + " " + board + " curriculum highlights"
                st.warning("Using minimal content as fallback...")
                
                try:
                    fallback_prompt = prompt.format(
                        context=fallback_content,
                        complexity_level=complexity_level,
                        num_questions=num_questions
                    )
                    response = llm.invoke(fallback_prompt)
                    fallback_answer = post_process_output(response.content)
                    
                    st.session_state.generated_questions = fallback_answer
                    return fallback_answer
                except:
                    st.error("Maximum retries reached. Please try again with less content.")
                    return "Failed to generate questions. Please try again with a smaller document or fewer questions."
    
    return "Failed to generate questions after multiple attempts."

def generate_answers(questions, board, class_level, subject, question_type):
    groq_api_key = get_api_key("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key is missing. Please check your secrets.")
        return "Error: Missing API key. Contact the administrator."
    
    llm = ChatGroq(api_key=groq_api_key, model='llama3-8b-8192')
    
    answer_prompt = get_answer_prompt(question_type, board, class_level, subject)
    
    total_questions_tokens = estimate_tokens(questions)
    if total_questions_tokens > 3000:
        st.warning("Many questions detected. Generating answers in batches...")
        
        question_parts = re.split(r'(\d+\.\s)', questions)
        all_questions = []
        current_question = ""
        
        for part in question_parts:
            if re.match(r'\d+\.\s', part):
                if current_question:
                    all_questions.append(current_question)
                current_question = part
            else:
                current_question += part
        
        if current_question:
            all_questions.append(current_question)
        
        batch_size = max(1, len(all_questions) // 2)
        batches = [all_questions[i:i+batch_size] for i in range(0, len(all_questions), batch_size)]
        
        all_answers = []
        for i, batch in enumerate(batches):
            batch_text = "".join(batch)
            st.info(f"Generating answers for batch {i+1} of {len(batches)}...")
            
            formatted_prompt = answer_prompt.format(questions=batch_text)
            
            try:
                response = llm.invoke(formatted_prompt)
                batch_answers = post_process_output(response.content)
                all_answers.append(batch_answers)
            except Exception as e:
                st.error(f"Error generating answers for batch {i+1}: {e}")
                all_answers.append(f"Error generating answers for questions {i*batch_size+1}-{min((i+1)*batch_size, len(all_questions))}")
        
        combined_answers = "\n\n".join(all_answers)
        cleaned_answers = post_process_output(combined_answers)
    else:
        formatted_prompt = answer_prompt.format(questions=questions)
        
        with st.spinner(f'AI is generating answers for {question_type} questions...'):
            start_time = time.time()
            response = llm.invoke(formatted_prompt)
            generation_time = time.time() - start_time
            st.success(f"Generated answers in {generation_time:.1f} seconds!")
        
        cleaned_answers = post_process_output(response.content)
    
    st.session_state.generated_answers = cleaned_answers
    
    return cleaned_answers

def post_process_output(text):
    patterns = [
        r'\n\s*1\.\s',
        r'\n\s*Question\s*1[\s\:\.]',
        r'\n\s*Q\.?\s*1[\s\:\.]',
    ]
    earliest_pos = len(text)
    for pattern in patterns:
        matches = re.search(pattern, text)
        if matches and matches.start() < earliest_pos:
            earliest_pos = matches.start()
    if earliest_pos < len(text):
        return text[earliest_pos+1:].strip()
    question_markers = re.search(r'\b(what|which|how|why|where|when|describe|explain|calculate|determine|identify|list)\b', text.lower())
    if question_markers:
        return text[question_markers.start():].strip()
    return text

def validate_questions(questions, board, class_level, subject):
    groq_api_key = get_api_key("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key is missing. Please check your secrets.")
        return questions
    
    llm = ChatGroq(api_key=groq_api_key, model='llama3-8b-8192')
    validation_prompt = f"""
    You are an expert validator for {board} Class {class_level} {subject} exam questions.
    
    Review the following questions and validate them based on these criteria:
    1. Relevance to {board} Class {class_level} curriculum
    2. Accuracy of content
    3. Clarity and unambiguous wording
    4. Appropriate difficulty level
    5. Grammatical correctness
    
    For each question, edit it directly if needed. Do not include reasoning or validation notes.
    Do not prefix the questions with "VALID:", "NEEDS REVISION:", or "REMOVE:".
    Simply output the corrected list of questions, starting with "1."
    
    Questions to validate:
    {questions}
    
    IMPORTANT FORMATTING INSTRUCTION: Output ONLY numbered questions starting with "1." - include ZERO preamble or explanations.
    Return the validated questions directly without any explanation of your validation process.
    """
    
    validation_tokens = estimate_tokens(validation_prompt)
    if validation_tokens > 4000:
        st.warning("Questions too long for full validation. Performing basic validation only.")
        max_question_tokens = 3000
        question_words = questions.split()
        truncated_questions = " ".join(question_words[:int(max_question_tokens/1.3)])
        
        validation_prompt = validation_prompt.replace(questions, truncated_questions)
    
    try:
        with st.spinner('Validating questions...'):
            response = llm.invoke(validation_prompt)
        return response.content
    except Exception as e:
        st.error(f"Validation error: {e}")
        return questions

def create_temp_pdf(content, board, class_level, subject, question_type, is_answer_key=False):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", style="B", size=16)
    
    def clean_text(text):
        if isinstance(text, str):
            return text.encode('latin-1', 'replace').decode('latin-1')
        return str(text)
    
    header_text = clean_text(f"{board} Class {class_level} - {subject} {question_type.title()} {'Answer Key' if is_answer_key else 'Questions'}")
    pdf.cell(200, 10, txt=header_text, ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", style="I", size=10)
    date_text = clean_text(f"Generated on: {datetime.now().strftime('%d-%m-%Y')}")
    pdf.cell(200, 10, txt=date_text, ln=True, align="R")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        clean_line = clean_text(line)
        
        if clean_line.strip().startswith(("Question", "Q.", "1.", "2.", "I.", "II.")):
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 7, txt=clean_line, ln=True, align="L")
            pdf.ln(2)
        elif clean_line.startswith(("A.", "B.", "C.", "D.", "a)", "b)", "c)", "d)", "(a)", "(b)")):
            pdf.set_font("Arial", size=12)
            pdf.cell(10, 5, txt="", ln=0)
            pdf.cell(0, 5, txt=clean_line, ln=True)
        else:
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 5, txt=clean_line)
            pdf.ln(1)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_path = temp_file.name
    
    pdf.output(temp_path)
    return temp_path

def main():
    st.set_page_config(
        page_title='CBSE Question Generator', 
        page_icon='📚',
        layout="wide"  
    )
    
    init_session_state()
    
    st.title("Indian Educational Board Question Generator 📚")
    
    st.markdown("""
    <style>
    .main-header {color: #1E88E5; font-size: 28px; font-weight: bold;}
    .sub-header {color: #0277BD; font-size: 20px; font-weight: bold;}
    .highlight {background-color: #f0f7fb; padding: 10px; border-radius: 5px; border-left: 5px solid #2196F3;}
    .success {background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;}
    .warning {background-color: #fff8e1; padding: 10px; border-radius: 5px; border-left: 5px solid #FF9800;}
    </style>
    """, unsafe_allow_html=True)
    
    groq_api_key = get_api_key("GROQ_API_KEY")
    
    if not groq_api_key:
        st.warning("GROQ API key is missing. The application may not function correctly.")
        st.info("For administrators: Please add GROQ_API_KEY to your Streamlit secrets.")
    
    with st.sidebar:
        st.header("Upload PDF and Settings")
        
        board = st.selectbox("Select Educational Board:", EDUCATIONAL_BOARDS)
        class_level = st.selectbox("Select Class:", CLASSES, index=9)
        subject = st.selectbox("Select Subject:", SUBJECTS)
        
        st.session_state.ocr_enabled = st.checkbox("Enable OCR for Scanned PDFs", value=True)
        
        st.markdown('<p class="highlight">Select a PDF file containing curriculum content. Both text-based and scanned PDFs are supported.</p>', unsafe_allow_html=True)
        upload_pdf = st.file_uploader(f"Upload {board} Class {class_level} {subject} PDF", type="pdf", accept_multiple_files=False)
        
        st.markdown("### Customize Question Generation")
        question_types = st.selectbox(
            "Select Question Type:",
            QUESTION_TYPES,
            index=0
        )
        complexity_level = st.selectbox(
            "Select Complexity Level:",
            ["basic", "intermediate", "advanced"],
            index=1
        )
        num_questions = st.slider(
            "Number of Questions:",
            min_value=5,
            max_value=25,
            value=10,
            step=1
        )
        validate = st.checkbox("Validate generated questions", value=True)
        
        st.markdown("### Performance Options")
        use_faster_model = st.checkbox("Use faster model (may reduce quality slightly)", value=True)
        
        # Question Bank in sidebar
        if st.session_state.question_bank:
            st.markdown("### 📚 Question Bank")
            st.metric("Saved Questions", len(st.session_state.question_bank))
            
            with st.expander("View Saved"):
                for entry in st.session_state.question_bank[-5:]:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.caption(f"{entry['settings']['subject']} - {entry['timestamp'][:10]}")
                    with col_b:
                        if st.button("📂", key=f"load_{entry['id']}"):
                            st.session_state.ai_answer = entry['questions']
                            st.session_state.answers = entry.get('answers', '')
                            st.session_state.answers_generated = bool(entry.get('answers'))
                            st.rerun()
        
        if upload_pdf:
            if st.button(f'Process {board} PDF', use_container_width=True):
                with st.spinner('Processing PDF...'):
                    # Load HuggingFace embeddings (no API key needed)
                    st.info("Using HuggingFace embeddings (no API key required)")
                    st.session_state.embedding = load_embeddings()
                    
                    st.session_state.raw_text = get_pdf_text(upload_pdf)
                    
                    if st.session_state.raw_text:
                        st.markdown("### Preview of Extracted Text")
                        preview_text = st.session_state.raw_text[:500] + "..." if len(st.session_state.raw_text) > 500 else st.session_state.raw_text
                        st.text_area("Extracted Text Preview", preview_text, height=100)
                        
                        split_chunks = split_data_into_chunks(st.session_state.raw_text)
                        st.session_state.vectors = store_chunks_vectorDB(chunks=split_chunks, embedding=st.session_state.embedding)
                        st.success('PDF processing complete! 🎉')
                        
                        word_count = len(st.session_state.raw_text.split())
                        st.markdown(f"<p class='success'>Extracted {word_count} words from the PDF.</p>", unsafe_allow_html=True)
                        
                        if 'answers' in st.session_state:
                            del st.session_state.answers
                        st.session_state.answers_generated = False
                    else:
                        st.error("Failed to extract text from the PDF. Please try another file.")
    
    if st.session_state.vectors:
        st.subheader(f"Generate {board} Class {class_level} {subject} Questions")
        
        if hasattr(st.session_state, 'raw_text'):
            word_count = len(st.session_state.raw_text.split())
            st.markdown(f"<p class='highlight'>Working with {word_count} words of content from your PDF.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Generate {question_types.title()} Questions", use_container_width=True):
                if 'answers' in st.session_state:
                    del st.session_state.answers
                st.session_state.answers_generated = False
                
                content = st.session_state.raw_text
                inputs = {
                    "content": content,
                    "question_types": question_types,
                    "complexity_level": complexity_level,
                    "num_questions": num_questions,
                    "board": board,
                    "class_level": class_level,
                    "subject": subject
                }
                
                st.session_state.ai_answer = generate_questions(inputs)
                
                if validate:
                    st.session_state.ai_answer = validate_questions(
                        st.session_state.ai_answer, 
                        board, 
                        class_level,
                        subject
                    )
                
                st.subheader(f"Generated {question_types.title()} Questions")
                st.markdown(f"<div class='success'>{st.session_state.ai_answer.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        
        with col2:
            if 'ai_answer' in st.session_state and not st.session_state.answers_generated:
                if st.button("Generate Answer Key", use_container_width=True):
                    with st.spinner("Generating detailed answers..."):
                        current_qt = question_types
                        if st.session_state.question_settings:
                            current_qt = st.session_state.question_settings.get("type", question_types)
                        
                        st.session_state.answers = generate_answers(
                            st.session_state.ai_answer,
                            board,
                            class_level,
                            subject,
                            current_qt
                        )
                        st.session_state.answers_generated = True
            
            if 'answers' in st.session_state and st.session_state.answers_generated:
                st.subheader("Answer Key")
                st.markdown(f"<div class='success'>{st.session_state.answers.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    
    # Enhanced question editing
    if 'ai_answer' in st.session_state:
        st.markdown("---")
        st.markdown("### ✏️ Edit Questions")
        
        edited_questions = st.text_area(
            "Review and edit your questions before exporting:",
            value=st.session_state.ai_answer,
            height=300,
            key="question_editor"
        )
        
        col_edit1, col_edit2, col_edit3 = st.columns(3)
        with col_edit1:
            if st.button("💾 Save Edits", use_container_width=True):
                st.session_state.ai_answer = edited_questions
                st.success("Questions updated!")
        
        with col_edit2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
        
        with col_edit3:
            # Save to question bank
            if st.button("📚 Save to Bank", use_container_width=True):
                if st.session_state.question_settings:
                    save_to_question_bank(
                        st.session_state.ai_answer,
                        st.session_state.get('answers', ''),
                        st.session_state.question_settings
                    )
                    st.success(f"Saved! Total in bank: {len(st.session_state.question_bank)}")
    
    # Enhanced export options with multiple formats
    if "ai_answer" in st.session_state:
        st.markdown("---")
        st.markdown("### 📥 Export Options")
        
        current_qt = question_types
        if st.session_state.question_settings:
            current_qt = st.session_state.question_settings.get("type", question_types)
        
        st.markdown("#### Export Questions")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📄 PDF", use_container_width=True, key="export_q_pdf"):
                try:
                    with st.spinner("Creating PDF..."):
                        temp_pdf_path = create_temp_pdf(
                            st.session_state.ai_answer,
                            board,
                            class_level,
                            subject,
                            current_qt,
                            is_answer_key=False
                        )
                        
                        with open(temp_pdf_path, "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        os.unlink(temp_pdf_path)
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_data,
                            file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_Questions.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="download_q_pdf"
                        )
                        st.success("PDF ready!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            if st.button("📝 Word", use_container_width=True, key="export_q_word"):
                try:
                    with st.spinner("Creating Word..."):
                        temp_docx_path = create_word_doc(
                            st.session_state.ai_answer,
                            board,
                            class_level,
                            subject,
                            current_qt,
                            is_answer_key=False
                        )
                        
                        with open(temp_docx_path, "rb") as docx_file:
                            docx_data = docx_file.read()
                        
                        os.unlink(temp_docx_path)
                        
                        st.download_button(
                            label="⬇️ Download Word",
                            data=docx_data,
                            file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_Questions.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                            key="download_q_word"
                        )
                        st.success("Word ready!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col3:
            if st.button("📊 Excel", use_container_width=True, key="export_q_excel"):
                try:
                    with st.spinner("Creating Excel..."):
                        temp_excel_path = create_excel_file(
                            st.session_state.ai_answer,
                            board,
                            class_level,
                            subject,
                            current_qt,
                            is_answer_key=False
                        )
                        
                        with open(temp_excel_path, "rb") as excel_file:
                            excel_data = excel_file.read()
                        
                        os.unlink(temp_excel_path)
                        
                        st.download_button(
                            label="⬇️ Download Excel",
                            data=excel_data,
                            file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_Questions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_q_excel"
                        )
                        st.success("Excel ready!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col4:
            if st.button("📋 JSON", use_container_width=True, key="export_q_json"):
                try:
                    json_data = create_json_export(
                        st.session_state.ai_answer,
                        st.session_state.question_settings or {}
                    )
                    
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_data,
                        file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_Questions.json",
                        mime="application/json",
                        use_container_width=True,
                        key="download_q_json"
                    )
                    st.success("JSON ready!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col5:
            if st.button("📝 Text", use_container_width=True, key="export_q_text"):
                st.download_button(
                    label="⬇️ Download Text",
                    data=st.session_state.ai_answer,
                    file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_Questions.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_q_text"
                )
                st.success("Text ready!")
        
        # Export answers if available
        if 'answers' in st.session_state and st.session_state.answers_generated:
            st.markdown("#### Export Answer Key")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("📄 PDF", use_container_width=True, key="export_a_pdf"):
                    try:
                        with st.spinner("Creating PDF..."):
                            temp_pdf_path = create_temp_pdf(
                                st.session_state.answers,
                                board,
                                class_level,
                                subject,
                                current_qt,
                                is_answer_key=True
                            )
                            
                            with open(temp_pdf_path, "rb") as pdf_file:
                                pdf_data = pdf_file.read()
                            
                            os.unlink(temp_pdf_path)
                            
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=pdf_data,
                                file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_AnswerKey.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="download_a_pdf"
                            )
                            st.success("PDF ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col2:
                if st.button("📝 Word", use_container_width=True, key="export_a_word"):
                    try:
                        with st.spinner("Creating Word..."):
                            temp_docx_path = create_word_doc(
                                st.session_state.answers,
                                board,
                                class_level,
                                subject,
                                current_qt,
                                is_answer_key=True
                            )
                            
                            with open(temp_docx_path, "rb") as docx_file:
                                docx_data = docx_file.read()
                            
                            os.unlink(temp_docx_path)
                            
                            st.download_button(
                                label="⬇️ Download Word",
                                data=docx_data,
                                file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_AnswerKey.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True,
                                key="download_a_word"
                            )
                            st.success("Word ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col3:
                if st.button("📊 Excel", use_container_width=True, key="export_a_excel"):
                    try:
                        with st.spinner("Creating Excel..."):
                            temp_excel_path = create_excel_file(
                                st.session_state.answers,
                                board,
                                class_level,
                                subject,
                                current_qt,
                                is_answer_key=True
                            )
                            
                            with open(temp_excel_path, "rb") as excel_file:
                                excel_data = excel_file.read()
                            
                            os.unlink(temp_excel_path)
                            
                            st.download_button(
                                label="⬇️ Download Excel",
                                data=excel_data,
                                file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_AnswerKey.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key="download_a_excel"
                            )
                            st.success("Excel ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col4:
                if st.button("📋 JSON", use_container_width=True, key="export_a_json"):
                    try:
                        settings_copy = st.session_state.question_settings.copy() if st.session_state.question_settings else {}
                        settings_copy['is_answer_key'] = True
                        
                        json_data = create_json_export(
                            st.session_state.answers,
                            settings_copy
                        )
                        
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_data,
                            file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_AnswerKey.json",
                            mime="application/json",
                            use_container_width=True,
                            key="download_a_json"
                        )
                        st.success("JSON ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col5:
                if st.button("📝 Text", use_container_width=True, key="export_a_text"):
                    st.download_button(
                        label="⬇️ Download Text",
                        data=st.session_state.answers,
                        file_name=f"{board}_Class{class_level}_{subject}_{current_qt}_AnswerKey.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_a_text"
                    )
                    st.success("Text ready!")
    
    if not st.session_state.vectors:
        st.markdown("## 🚀 How to Get Started")
        st.markdown("""
        1. **Upload a PDF**: Select a PDF containing curriculum material for your chosen subject
        2. **Click 'Process PDF'**: The system will extract text from your PDF (supports both text PDFs and scanned/image PDFs)
        3. **Configure Questions**: Choose the question type, complexity level, and number of questions
        4. **Generate Questions**: Click the button to create educational board-specific questions
        5. **Create Answer Key**: Generate detailed answers for the questions
        6. **Export**: Download in multiple formats - PDF, Word, Excel, JSON, or Text
        7. **Save to Bank**: Store frequently used questions for future reference
        """)
        
        st.info("This tool supports OCR for scanned PDFs and handwritten content. Enable the OCR option in the sidebar if you're using scanned materials.")
        
        st.success("✨ Now using HuggingFace embeddings - No Google API payment required!")

if __name__ == "__main__":
    main()
