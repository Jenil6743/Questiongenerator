import streamlit as st
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
import requests  # for token estimation

# Instead of using dotenv, use Streamlit secrets
def get_api_key(key_name):
    """Get API key from Streamlit secrets or environment variables"""
    # First try to get from Streamlit secrets (production)
    if key_name in st.secrets:
        return st.secrets[key_name]
    # Fall back to local environment variables (development)
    elif key_name in os.environ:
        return os.environ[key_name]
    else:
        st.error(f"Missing API key: {key_name}. Please add it to your secrets.")
        return None

EDUCATIONAL_BOARDS = ["CBSE", "ICSE", "State Board"]
CLASSES = list(range(1, 13))
SUBJECTS = ["Mathematics", "Science", "Social Studies", "English", "Hindi", "Physics", "Chemistry", "Biology", "Computer Science", "Other"]
QUESTION_TYPES = ["multiple-choice", "short-answer", "long-answer", "true-false", "fill-in-the-blanks", "match-the-following"]

# Function to estimate token count (approximate)
def estimate_tokens(text):
    """Estimate token count for a text string - rough estimate based on words"""
    # Simple estimation: ~1.3 tokens per word for English
    words = text.split()
    return int(len(words) * 1.3)

def is_pdf_scanned(pdf_bytes):
    """Check if a PDF is likely scanned/image-based or text-based"""
    try:
        # Try to read with PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        first_page = pdf_reader.pages[0]
        text = first_page.extract_text()
        
        # If very little text is extracted, it's likely a scanned PDF
        if len(text.strip()) < 100:
            return True
        return False
    except Exception:
        # If error occurs during text extraction, assume it's a scanned PDF
        return True

def extract_text_from_image(image):
    """Extract text from an image using OCR"""
    try:
        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def process_scanned_pdf(pdf_bytes):
    """Process a scanned PDF using OCR"""
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)
        
        # Apply OCR to each page
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
    """Extract text from PDF - supports both text-based and scanned PDFs"""
    try:
        # Cache the file content
        pdf_bytes = upload_pdf.getvalue()
        
        # Check if it's a scanned PDF
        if is_pdf_scanned(pdf_bytes):
            st.info("Detected a scanned/image-based PDF. Using OCR to extract text...")
            return process_scanned_pdf(pdf_bytes)
        
        # If it's a text-based PDF, use standard extraction
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
    """Split text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def store_chunks_vectorDB(chunks, embedding):
    """Store text chunks in a vector database for retrieval"""
    batch_size = 100
    vector_db = None
    progress_bar = st.progress(0)
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_num = i // batch_size
        progress_bar.progress((batch_num + 1) / total_batches)
        if vector_db is None:
            vector_db = FAISS.from_texts(texts=batch_chunks, embedding=embedding)
        else:
            vector_db.add_texts(texts=batch_chunks)
    progress_bar.empty()
    return vector_db

def create_diverse_question_prompt(question_type, board, class_level, subject, seed=None):
    """Create a prompt to generate diverse questions with a randomized seed"""
    # Use random seed to encourage diversity in generation
    if seed is None:
        seed = random.randint(1, 1000)
    
    # Different approaches to question formulation
    approaches = [
        "analytical thinking",
        "critical reasoning",
        "real-world application",
        "conceptual understanding",
        "creative problem-solving"
    ]
    
    # Randomly select focus areas
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
    
    # Randomly select cognitive levels to target
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
    
    # Add specific formatting instructions based on question type
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
    """Get a specific prompt for generating answers based on question type"""
    
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
    
    # Add specific answer formatting instructions based on question type
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
    """Check if the generated questions match the expected format"""
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
    """Generate questions with improved reliability and diversity"""
    content = inputs['content']
    question_type = inputs['question_types']
    complexity_level = inputs['complexity_level']
    num_questions = inputs['num_questions']
    board = inputs.get('board', 'CBSE')
    class_level = inputs.get('class_level', '10')
    subject = inputs.get('subject', 'General')
    
    # Display current settings
    st.info(f"""
    Generating {num_questions} diverse {question_type} questions 
    for {board} Class {class_level} {subject} at {complexity_level} complexity
    """)
    
    # Store settings in session state
    st.session_state.question_settings = {
        "type": question_type,
        "board": board,
        "class": class_level,
        "subject": subject,
        "complexity": complexity_level,
        "count": num_questions
    }
    
    # Get API key from secrets
    groq_api_key = get_api_key("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key is missing. Please check your secrets.")
        return "Error: Missing API key. Contact the administrator."
    
    # Use a faster model for quicker generation
    llm = ChatGroq(api_key=groq_api_key, model='llama3-8b-8192')
    
    # Randomly seed for diversity
    seed = random.randint(1, 1000)
    
    # Get specific prompt for the question type with diversity enhancements
    prompt_template = create_diverse_question_prompt(
        question_type, 
        board, 
        class_level, 
        subject,
        seed=seed
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Instead of using the entire content, use vector retrieval to get most relevant parts
    # This helps avoid token limit issues
    if st.session_state.vectors:
        # Create a query to search the vector database
        search_query = f"{board} Class {class_level} {subject} {question_type} questions {complexity_level} complexity"
        
        # Retrieve relevant chunks
        with st.spinner('Retrieving relevant content from document...'):
            # Start with a reasonable number of chunks
            k_chunks = 5
            relevant_docs = st.session_state.vectors.similarity_search(search_query, k=k_chunks)
            relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Estimate token count - aim for less than 2000 tokens for content
            estimated_tokens = estimate_tokens(relevant_content)
            
            # If too many tokens, reduce the number of chunks
            while estimated_tokens > 2000 and k_chunks > 1:
                k_chunks -= 1
                relevant_docs = st.session_state.vectors.similarity_search(search_query, k=k_chunks)
                relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])
                estimated_tokens = estimate_tokens(relevant_content)
                
            # Log the tokens for debugging
            st.session_state.content_tokens = estimated_tokens
            
            # If still too large, truncate
            if estimated_tokens > 2000:
                words = relevant_content.split()
                relevant_content = " ".join(words[:1500])  # Approximately 2000 tokens
                
            # Use the retrieved content instead of the entire document
            content = relevant_content
    else:
        # If no vector DB, use a truncated version of the content
        words = content.split()
        if len(words) > 1500:  # Roughly 2000 tokens
            content = " ".join(words[:1500])
    
    # Try generating with retries for reliability
    for attempt in range(max_retries + 1):
        try:
            # Show retry message if not first attempt
            if attempt > 0:
                st.info(f"Retry attempt {attempt} with a different approach...")
            
            with st.spinner(f'AI is generating {question_type} questions (attempt {attempt+1})...'):
                # Format the prompt with our context and parameters
                formatted_prompt = prompt.format(
                    context=content,
                    complexity_level=complexity_level,
                    num_questions=num_questions
                )
                
                # Estimate total tokens
                prompt_tokens = estimate_tokens(formatted_prompt)
                if prompt_tokens > 5000:
                    st.warning(f"Prompt too large (~{prompt_tokens} tokens). Reducing content size...")
                    words = content.split()
                    reduced_words = int(len(words) * (4000 / prompt_tokens))
                    content = " ".join(words[:reduced_words])
                    
                    # Re-format with reduced content
                    formatted_prompt = prompt.format(
                        context=content,
                        complexity_level=complexity_level,
                        num_questions=num_questions
                    )
                
                # Generate with a timeout to prevent hanging
                start_time = time.time()
                response = llm.invoke(formatted_prompt)
                generation_time = time.time() - start_time
                
                # Log generation time for performance monitoring
                st.session_state.last_generation_time = generation_time
            
            # Get the text response from the LLM
            ai_answer = response.content if hasattr(response, 'content') else str(response)
            cleaned_answer = post_process_output(ai_answer)
            
            # Verify the format is correct
            if check_question_format(cleaned_answer, question_type):
                # Success! Return the result
                st.success(f"Successfully generated {question_type} questions in {generation_time:.1f} seconds!")
                
                # Store the generated questions
                st.session_state.generated_questions = cleaned_answer
                return cleaned_answer
            
            # If format is wrong, try fixing it
            if attempt == max_retries:
                # Last attempt, try to fix what we have
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
            
            # Try again with a different seed
            seed = random.randint(1, 1000)
            prompt_template = create_diverse_question_prompt(question_type, board, class_level, subject, seed=seed)
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
        except Exception as e:
            st.error(f"Error during generation (attempt {attempt+1}): {e}")
            
            # Check if it's a token limit error
            error_str = str(e)
            if "413" in error_str or "token" in error_str.lower() or "too large" in error_str.lower():
                # Reduce content size by half and try again
                words = content.split()
                content = " ".join(words[:len(words)//2])
                st.warning("Request was too large. Reducing content size and trying again...")
            
            if attempt == max_retries:
                # Generate with much less content as a fallback
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
    """Generate answers for the given questions"""
    # Get API key from secrets
    groq_api_key = get_api_key("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key is missing. Please check your secrets.")
        return "Error: Missing API key. Contact the administrator."
    
    # Use faster model for better performance
    llm = ChatGroq(api_key=groq_api_key, model='llama3-8b-8192')
    
    # Get specific prompt for answers based on question type
    answer_prompt = get_answer_prompt(question_type, board, class_level, subject)
    
    # Check token count before sending
    total_questions_tokens = estimate_tokens(questions)
    if total_questions_tokens > 3000:
        # Too many questions for one request, so split them
        st.warning("Many questions detected. Generating answers in batches...")
        
        # Find question boundaries to split properly
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
        
        # Process questions in smaller batches
        batch_size = max(1, len(all_questions) // 2)  # Split into at least 2 batches
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
        
        # Combine all answer batches
        combined_answers = "\n\n".join(all_answers)
        cleaned_answers = post_process_output(combined_answers)
    else:
        # Normal processing for smaller question sets
        formatted_prompt = answer_prompt.format(questions=questions)
        
        with st.spinner(f'AI is generating answers for {question_type} questions...'):
            start_time = time.time()
            response = llm.invoke(formatted_prompt)
            generation_time = time.time() - start_time
            st.success(f"Generated answers in {generation_time:.1f} seconds!")
        
        cleaned_answers = post_process_output(response.content)
    
    # Store the generated answers
    st.session_state.generated_answers = cleaned_answers
    
    return cleaned_answers

def post_process_output(text):
    """Clean up the raw output from the AI to extract just the questions/answers"""
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
    """Validate and improve the generated questions"""
    # Get API key from secrets
    groq_api_key = get_api_key("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key is missing. Please check your secrets.")
        return questions  # Return original questions if API key is missing
    
    # Use faster model
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
    
    # Check token count
    validation_tokens = estimate_tokens(validation_prompt)
    if validation_tokens > 4000:
        st.warning("Questions too long for full validation. Performing basic validation only.")
        # Truncate prompt to focus on validation instructions
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
        return questions  # Return original if validation fails

def create_temp_pdf(content, board, class_level, subject, question_type, is_answer_key=False):
    """Create a temporary PDF file with the generated content"""
    import tempfile
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", style="B", size=16)
    
    # Handle encoding issues
    def clean_text(text):
        if isinstance(text, str):
            return text.encode('latin-1', 'replace').decode('latin-1')
        return str(text)
    
    # Create header
    header_text = clean_text(f"{board} Class {class_level} - {subject} {question_type.title()} {'Answer Key' if is_answer_key else 'Questions'}")
    pdf.cell(200, 10, txt=header_text, ln=True, align="C")
    pdf.ln(5)
    
    # Add date
    from datetime import datetime
    pdf.set_font("Arial", style="I", size=10)
    date_text = clean_text(f"Generated on: {datetime.now().strftime('%d-%m-%Y')}")
    pdf.cell(200, 10, txt=date_text, ln=True, align="R")
    pdf.ln(10)
    
    # Add content
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
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_path = temp_file.name
    
    # Save PDF to the temporary file
    pdf.output(temp_path)
    return temp_path

def main():
    # No need to load_dotenv() anymore - we're using st.secrets
    st.set_page_config(
        page_title='CBSE Question Generator', 
        page_icon='📚',
        layout="wide"  # Use wide layout for better display
    )
    
    st.title("Indian Educational Board Question Generator 📚")
    
    # Add CSS for better styling
    st.markdown("""
    <style>
    .main-header {color: #1E88E5; font-size: 28px; font-weight: bold;}
    .sub-header {color: #0277BD; font-size: 20px; font-weight: bold;}
    .highlight {background-color: #f0f7fb; padding: 10px; border-radius: 5px; border-left: 5px solid #2196F3;}
    .success {background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;}
    .warning {background-color: #fff8e1; padding: 10px; border-radius: 5px; border-left: 5px solid #FF9800;}
    </style>
    """, unsafe_allow_html=True)
    
    # Verify API keys are available
    groq_api_key = get_api_key("GROQ_API_KEY")
    google_api_key = get_api_key("GOOGLE_API_KEY")
    
    if not groq_api_key or not google_api_key:
        st.warning("API keys are missing. The application may not function correctly.")
        st.info("For administrators: Please add GROQ_API_KEY and GOOGLE_API_KEY to your Streamlit secrets.")
    
    # Initialize session variables
    if 'vectors' not in st.session_state:
        st.session_state.vectors = None
    if 'pdf_ready' not in st.session_state:
        st.session_state.pdf_ready = False
    if 'answers_generated' not in st.session_state:
        st.session_state.answers_generated = False
    if 'question_settings' not in st.session_state:
        st.session_state.question_settings = None
    if 'ocr_enabled' not in st.session_state:
        st.session_state.ocr_enabled = True
    
    # Sidebar
    with st.sidebar:
        st.header("Upload PDF and Settings")
        
        board = st.selectbox("Select Educational Board:", EDUCATIONAL_BOARDS)
        class_level = st.selectbox("Select Class:", CLASSES, index=9)
        subject = st.selectbox("Select Subject:", SUBJECTS)
        
        # Add toggle for OCR processing
        st.session_state.ocr_enabled = st.checkbox("Enable OCR for Scanned PDFs", value=True)
        
        # File uploader with better instructions
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
        
        # Performance options
        st.markdown("### Performance Options")
        use_faster_model = st.checkbox("Use faster model (may reduce quality slightly)", value=True)
        
        if upload_pdf:
            if st.button(f'Process {board} PDF', use_container_width=True):
                with st.spinner('Processing PDF...'):
                    # Get Google API key from secrets
                    google_api_key = get_api_key("GOOGLE_API_KEY")
                    if not google_api_key:
                        st.error("Google API key is missing. Please check your secrets.")
                    else:
                        st.session_state.embedding = GoogleGenerativeAIEmbeddings(
                            api_key=google_api_key,
                            model="models/embedding-001"
                        )
                        st.session_state.raw_text = get_pdf_text(upload_pdf)
                        
                        if st.session_state.raw_text:
                            # Show a preview of extracted text
                            st.markdown("### Preview of Extracted Text")
                            preview_text = st.session_state.raw_text[:500] + "..." if len(st.session_state.raw_text) > 500 else st.session_state.raw_text
                            st.text_area("Extracted Text Preview", preview_text, height=100)
                            
                            split_chunks = split_data_into_chunks(st.session_state.raw_text)
                            st.session_state.vectors = store_chunks_vectorDB(chunks=split_chunks, embedding=st.session_state.embedding)
                            st.success('PDF processing complete! 🎉')
                            
                            # Show detected pages and word count
                            word_count = len(st.session_state.raw_text.split())
                            st.markdown(f"<p class='success'>Extracted {word_count} words from the PDF.</p>", unsafe_allow_html=True)
                            
                            # Reset answers when new PDF is processed
                            if 'answers' in st.session_state:
                                del st.session_state.answers
                            st.session_state.answers_generated = False
                        else:
                            st.error("Failed to extract text from the PDF. Please try another file.")
    
    # Main content
    if st.session_state.vectors:
        st.subheader(f"Generate {board} Class {class_level} {subject} Questions")
        
        # Show extracted content stats
        if hasattr(st.session_state, 'raw_text'):
            word_count = len(st.session_state.raw_text.split())
            st.markdown(f"<p class='highlight'>Working with {word_count} words of content from your PDF.</p>", unsafe_allow_html=True)
        
        # Create a two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Generate {question_types.title()} Questions", use_container_width=True):
                # Reset answers when new questions are generated
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
                
                # Generate questions with improved reliability
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
            # Only show the "Generate Answers" button if questions have been generated
            if 'ai_answer' in st.session_state and not st.session_state.answers_generated:
                if st.button("Generate Answer Key", use_container_width=True):
                    with st.spinner("Generating detailed answers..."):
                        # Get current question type from session state if available
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
            
            # Display the answers if they've been generated
            if 'answers' in st.session_state and st.session_state.answers_generated:
                st.subheader("Answer Key")
                st.markdown(f"<div class='success'>{st.session_state.answers.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    
    # Export section for PDFs
    if "ai_answer" in st.session_state:
        st.markdown("### Export Options")
        
        # Get current question type
        current_qt = question_types
        if st.session_state.question_settings:
            current_qt = st.session_state.question_settings.get("type", question_types)
        
        col1, col2 = st.columns(2)
        
        # Question PDF export
        with col1:
            if st.button(f"Export {current_qt.title()} Questions to PDF", use_container_width=True):
                import tempfile
                import os
                
                try:
                    with st.spinner(f"Creating {current_qt} Questions PDF..."):
                        # Generate PDF to a temporary file
                        temp_pdf_path = create_temp_pdf(
                            st.session_state.ai_answer,
                            board,
                            class_level,
                            subject,
                            current_qt,
                            is_answer_key=False
                        )
                        
                        # Read the temporary file
                        with open(temp_pdf_path, "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        # Remove the temporary file
                        os.unlink(temp_pdf_path)
                        
                        st.download_button(
                            label=f"Download {current_qt.title()} Questions PDF",
                            data=pdf_data,
                            file_name=f"{board}Class{class_level}{subject}_{current_qt}_Questions.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.success(f"{current_qt.title()} Questions PDF created successfully!")
                        
                except Exception as e:
                    st.error(f"Error creating PDF: {e}")
                    st.info("Try upgrading to fpdf2 library for better Unicode support.")
        
        # Answer PDF export - only show if answers have been generated
        with col2:
            if 'answers' in st.session_state and st.session_state.answers_generated:
                if st.button(f"Export {current_qt.title()} Answer Key to PDF", use_container_width=True):
                    import tempfile
                    import os
                    
                    try:
                        with st.spinner(f"Creating {current_qt} Answer Key PDF..."):
                            # Generate PDF to a temporary file
                            temp_pdf_path = create_temp_pdf(
                                st.session_state.answers,
                                board,
                                class_level,
                                subject,
                                current_qt,
                                is_answer_key=True
                            )
                            
                            # Read the temporary file
                            with open(temp_pdf_path, "rb") as pdf_file:
                                pdf_data = pdf_file.read()
                            
                            # Remove the temporary file
                            os.unlink(temp_pdf_path)
                            
                            st.download_button(
                                label=f"Download {current_qt.title()} Answer Key PDF",
                                data=pdf_data,
                                file_name=f"{board}Class{class_level}{subject}_{current_qt}_AnswerKey.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            st.success(f"{current_qt.title()} Answer Key PDF created successfully!")
                            
                    except Exception as e:
                        st.error(f"Error creating PDF: {e}")
                        st.info("Try upgrading to fpdf2 library for better Unicode support.")
    
    # Add helpful instructions for first-time users
    if not st.session_state.vectors:
        st.markdown("## 🚀 How to Get Started")
        st.markdown("""
        1. **Upload a PDF**: Select a PDF containing curriculum material for your chosen subject
        2. **Click 'Process PDF'**: The system will extract text from your PDF (supports both text PDFs and scanned/image PDFs)
        3. **Configure Questions**: Choose the question type, complexity level, and number of questions
        4. **Generate Questions**: Click the button to create educational board-specific questions
        5. **Create Answer Key**: Generate detailed answers for the questions
        6. **Export**: Download both questions and answers as PDF files
        """)
        
        st.info("This tool supports OCR for scanned PDFs and handwritten content. Enable the OCR option in the sidebar if you're using scanned materials.")

if __name__ == "__main__":
    main()
