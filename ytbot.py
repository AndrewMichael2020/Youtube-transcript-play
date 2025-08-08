# Import necessary libraries for the YouTube bot
import os
import requests
import gradio as gr
import re  #For extracting video id 
from dotenv import load_dotenv  # For loading environment variables
from youtube_transcript_api import YouTubeTranscriptApi, _errors  # For extracting transcripts from YouTube videos
try:
    from youtube_transcript_api.proxies import ProxyConfig, GenericProxyConfig
except Exception:
    ProxyConfig = None
    GenericProxyConfig = None
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI LLM and embeddings via LangChain
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates
# Default fallback transcript text (ignored unless user edits/replaces it)
DEFAULT_FALLBACK_TRANSCRIPT = (
    "FALLBACK TRANSCRIPT (replace this with a real transcript if desired)\n\n"
    "Text: In this tutorial, we cover Python basics including variables, loops, and functions. Start: 0.0\n"
    "Text: We then build a small project to practice these concepts step by step. Start: 45.2\n"
    "Text: Finally, we summarize key takeaways and provide tips for next steps. Start: 120.7\n"
)



def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(url):
    video_id = get_video_id(url)
    # No cookies are used; rely solely on public transcript availability
    try:
        # Optional cookie usage: we will only load a minimal subset of cookies if the file exists
        cookies_path = os.path.join(os.path.dirname(__file__), "youtube_cookies.txt")
        http_client = None
        cookies_loaded = False
        if os.path.exists(cookies_path):
            # Minimal set of cookies to keep/use
            keep = {
                "SID", "HSID", "SSID", "APISID", "SAPISID",
                "__Secure-1PSID", "__Secure-3PSID", "__Secure-1PAPISID", "__Secure-3PAPISID",
                "LOGIN_INFO", "PREF", "VISITOR_INFO1_LIVE", "YSC"
            }
            sess = requests.Session()
            # Prevent env proxies from being used when cookies are present
            sess.trust_env = False
            sess.proxies = {}
            # Parse Netscape cookie file and also trim the file to only kept cookies
            header = "# Netscape HTTP Cookie File\n"
            kept_lines = [header]
            with open(cookies_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        # Keep only a single header in output
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 7:
                        continue
                    domain, include_sub, path, secure, expires, name, value = parts[-7:]
                    if name in keep and domain.endswith("youtube.com"):
                        sess.cookies.set(name, value, domain=domain, path=path)
                        kept_lines.append("\t".join([domain, include_sub, path, secure, expires, name, value]) + "\n")
            # Rewrite cookie file with only essential cookies
            try:
                with open(cookies_path, "w", encoding="utf-8") as wf:
                    wf.writelines(kept_lines)
            except Exception:
                # Non-fatal: proceed even if we can't rewrite
                pass
            http_client = sess
            cookies_loaded = True

        # Only use proxies if cookies are NOT provided
        proxy_cfg = None
        if not cookies_loaded:
            http_proxy = os.getenv("YT_HTTP_PROXY") or os.getenv("HTTP_PROXY")
            https_proxy = os.getenv("YT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY")
            if GenericProxyConfig and (http_proxy or https_proxy):
                proxy_cfg = GenericProxyConfig(http_url=http_proxy, https_url=https_proxy)

        if proxy_cfg and http_client:
            ytt_api = YouTubeTranscriptApi(proxy_config=proxy_cfg, http_client=http_client)
        elif proxy_cfg:
            ytt_api = YouTubeTranscriptApi(proxy_config=proxy_cfg)
        elif http_client:
            ytt_api = YouTubeTranscriptApi(http_client=http_client)
        else:
            ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        # Prefer manually created English transcript; fallback to generated English
        transcript_obj = None
        try:
            transcript_obj = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            try:
                transcript_obj = transcript_list.find_generated_transcript(["en"])
            except Exception:
                transcript_obj = None

        if transcript_obj is None:
            return None

        return transcript_obj.fetch()
    except Exception as e:
        print(f"Transcript fetch failed: {e}")
        return None

def process(transcript):
    if not transcript:
        return ""
    txt = ""
    for i in transcript:
        try:
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except KeyError:
            pass
    return txt


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


def setup_credentials():
    """
    Load environment variables (e.g., OPENAI_API_KEY). Kept signature compatible with previous IBM setup
    to avoid changing call sites. Returns placeholders to preserve logic.
    """
    load_dotenv()
    # Placeholders to keep call sites unchanged
    model_id = None
    credentials = None
    client = None
    project_id = None
    return model_id, credentials, client, project_id

def define_parameters():
    # Return a dictionary containing the parameters for the OpenAI model
    return {
        "temperature": 0,
        "max_tokens": 900,
    }


def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    """
    Return an OpenAI chat model via LangChain. Signature retained to avoid changing callers.
    """
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=parameters.get("temperature", 0),
        max_tokens=parameters.get("max_tokens", 900),
    )



def setup_embedding_model(credentials, project_id):
    """
    Create and return an instance of OpenAIEmbeddings. Signature retained to avoid changing callers.
    """
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))



def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)



def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
    
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results


def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
    System: You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.

    User: Please summarize the following YouTube video transcript:

    {transcript}
    Assistant:
    """
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt


def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
    
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    # Define the template string
    qa_template = """
    System: You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.

    User:
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    {question}
    Assistant:
    """
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)

    # Generate answer using the QA chain
    answer = qa_chain.predict(context=relevant_context, question=question)

    return answer


# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""


def summarize_video(video_url, pasted_transcript=""):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    
    if pasted_transcript and pasted_transcript.strip() and pasted_transcript.strip() != DEFAULT_FALLBACK_TRANSCRIPT.strip():
        processed_transcript = pasted_transcript.strip()
    elif video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Step 1: Set up IBM Watson credentials
        model_id, credentials, client, project_id = setup_credentials()

        # Step 2: Initialize WatsonX LLM for summarization
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."


def answer_question(video_url, user_question, pasted_transcript=""):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if pasted_transcript and pasted_transcript.strip() and pasted_transcript.strip() != DEFAULT_FALLBACK_TRANSCRIPT.strip():
            processed_transcript = pasted_transcript.strip()
        elif video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL or paste a transcript."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up IBM Watson credentials
        model_id, credentials, client, project_id = setup_credentials()

        # Step 3: Initialize WatsonX LLM for Q&A
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model(credentials, project_id)
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."



with gr.Blocks() as interface:

    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )

    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Optional manual transcript paste (fallback when fetching is blocked)
    pasted_transcript = gr.Textbox(label="Paste transcript (optional)", lines=10, placeholder="Paste full transcript here to bypass fetching", value=DEFAULT_FALLBACK_TRANSCRIPT)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=[video_url, pasted_transcript], outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input, pasted_transcript], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="0.0.0.0", server_port=7860)