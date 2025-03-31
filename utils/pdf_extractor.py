import os
import PyPDF2
import re


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """
    text = ""

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"

        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip()

        return text

    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def process_all_pdfs(pdf_dir, output_dir):
    """
    Process all PDFs in a directory and save the extracted text to output files.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save processed text files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        output_file = os.path.join(output_dir, pdf_file.replace('.pdf', '.txt'))

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)

        if text:
            # Save to text file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"Processed {pdf_file} -> {output_file}")
        else:
            print(f"Failed to process {pdf_file}")