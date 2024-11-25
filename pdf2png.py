import fitz  # PyMuPDF
import os

def pdf_to_images(pdf_path, output_dir, dpi=300):
    """
    Convert a PDF file to images.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output images.
        dpi (int): DPI (dots per inch) for the output images (default: 300).
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Iterate through each page in the PDF
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document[page_num]
        
        # Render the page to a high-resolution image
        zoom = dpi / 72  # Zoom factor (default PDF DPI is 72)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)
        
        # Save the image
        output_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pixmap.save(output_path)
        print(f"Saved page {page_num + 1} as {output_path}")
    
    print("PDF conversion to images completed.")
    pdf_document.close()

# Example Usage
pdf_path = "static\images\Level3_baseline_radar.pdf"  # Path to the PDF file
output_dir = "static\images\Level3_baseline_radar.png"  # Directory to save the images
pdf_to_images(pdf_path, output_dir)
