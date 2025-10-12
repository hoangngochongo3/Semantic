import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
            title=text.splitlines()[0]
    lines = text.splitlines()
    title = lines[0] if lines else ""
    text_without_first_line = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return title,text_without_first_line
