import streamlit as st
from PIL import Image
import numpy as np

def encode_text_in_image(image_path, text_to_hide, output_path="encoded_image.png"):
    """Encodes text into an image using LSB steganography with a null terminator."""
    img = Image.open(image_path).convert('RGB')
    text_to_hide_with_terminator = text_to_hide + '\0' 
    binary_text = ''.join(format(ord(char), '08b') for char in text_to_hide_with_terminator) 

    if len(binary_text) > img.width * img.height * 3: 
        raise ValueError("Text is too long to hide in this image.")

    img_array = np.array(img)
    binary_index = 0

    for row in range(img_array.shape[0]):
        for col in range(img_array.shape[1]):
            for color_channel in range(3):
                if binary_index < len(binary_text):
                    pixel_value = img_array[row, col, color_channel]
                    lsb = pixel_value & 1
                    text_bit = int(binary_text[binary_index])

                    if lsb != text_bit: # Only change if LSB is different from text bit
                        if text_bit == 1:
                            if pixel_value % 2 == 0: # Ensure pixel value becomes odd
                                pixel_value += 1
                        else:
                            if pixel_value % 2 != 0: # Ensure pixel value becomes even
                                pixel_value -= 1

                    img_array[row, col, color_channel] = pixel_value
                    binary_index += 1

    encoded_img = Image.fromarray(img_array)
    encoded_img.save(output_path)
    return output_path

def decode_text_from_image(image_path):
    """Decodes hidden text from an image using LSB steganography, stopping at null terminator."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    binary_message = ""
    decoded_text = ""
    binary_char = ""

    for row in range(img_array.shape[0]):
        for col in range(img_array.shape[1]):
            for color_channel in range(3):
                binary_message += str(img_array[row, col, color_channel] & 1) # Extract LSB
                if len(binary_message) % 8 == 0: # 8 bits = 1 character
                    binary_char = binary_message[-8:] # Get the last 8 bits
                    if binary_char == '00000000': # Null terminator (stop decoding here)
                        return decoded_text
                    try:
                        decoded_text += chr(int(binary_char, 2)) # Binary to character
                    except ValueError: 
                        pass 

    return decoded_text 


# --- Streamlit UI ---
st.title("Secret Message in a Picture: Image Steganography")
st.write("Hide secret messages inside images using Least Significant Bit (LSB) steganography.")

operation_type = st.radio("Choose Operation:", ["Encode (Hide Text)", "Decode (Reveal Text)"])

uploaded_image = st.file_uploader("Upload Image:", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if operation_type == "Encode (Hide Text)":
        text_to_encode = st.text_area("Enter Secret Message to Hide:")
        if st.button("Encode and Download"):
            if text_to_encode:
                try:
                    encoded_image_path = encode_text_in_image(uploaded_image, text_to_encode)
                    with open(encoded_image_path, "rb") as file:
                        st.download_button(
                            label="Download Encoded Image",
                            data=file,
                            file_name="encoded_image.png",
                            mime="image/png"
                        )
                    st.success("Text encoded successfully! Download the 'encoded_image.png' file.")
                except ValueError as e:
                    st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"An error occurred during encoding: {e}")
            else:
                st.warning("Please enter a secret message to hide.")

    elif operation_type == "Decode (Reveal Text)":
        if st.button("Decode and Reveal Text"):
            try:
                decoded_message = decode_text_from_image(uploaded_image)
                st.subheader("Decoded Secret Message:")
                if decoded_message:
                    st.write(decoded_message)
                else:
                    st.info("No secret message found or image is not encoded with this method.")
            except Exception as e:
                st.error(f"An error occurred during decoding: {e}")

st.sidebar.header("About Steganography")
st.sidebar.info(
    "Steganography is like hiding a secret message in plain sight.  Imagine hiding a letter inside a book instead of writing it on the cover."
    "This app uses a simple trick called LSB steganography to hide your secret text inside a picture."
    "It changes the picture in a tiny way that people can't normally see, but your message is secretly stored there."
    "This updated app is smarter now! It knows exactly where your secret message ends, so it doesn't show any extra nonsense at the end."
    "It's a basic example, but it shows how you can hide information in a way that's not obvious."
)
st.sidebar.warning("Remember: This is just a simple example to learn from. Real secret hiding techniques can be much more complicated.")

