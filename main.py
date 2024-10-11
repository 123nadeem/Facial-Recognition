# import face_recognition
# from PIL import Image, ImageDraw
# import numpy as np
# import os
# import pickle
# import json  # Import the json module

# embeddings_file = "models/known_faces_embeddings.pkl"

# # Initialize known face encodings and names
# known_face_encodings = []
# known_face_names = []

# # Load known faces if the embeddings file exists
# if os.path.exists(embeddings_file):
#     with open(embeddings_file, "rb") as f:
#         known_face_encodings, known_face_names = pickle.load(f)
# else:
#     print("No known faces data found. Starting with empty lists.")

# def add_new_face(face_encoding, name):
#     """
#     Adds a new face encoding and its name to the known faces list
#     and saves the updated data to the embeddings file.
#     """
#     known_face_encodings.append(face_encoding)
#     known_face_names.append(name)
#     with open(embeddings_file, "wb") as f:
#         pickle.dump((known_face_encodings, known_face_names), f)

# def process_image(input_image_path, output_image_path):
#     """
#     Processes the input image to detect and recognize faces,
#     draws rectangles and names on recognized faces, and saves the output image.
#     """
#     if not os.path.exists(input_image_path):
#         print(f"Input image file does not exist: {input_image_path}")
#         return {"names": []}

#     image_file = Image.open(input_image_path)
#     image = np.array(image_file)

#     face_locations = face_recognition.face_locations(image, model="cnn")
#     face_encodings = face_recognition.face_encodings(image, face_locations)

#     if not face_locations:
#         print("No faces detected in the image.")
#         return {"names": []}

#     pil_image = Image.fromarray(image)
#     draw = ImageDraw.Draw(pil_image)

#     names = []
#     face_info = []

#     base_name = os.path.basename(input_image_path)
#     name_from_filename = os.path.splitext(base_name)[0]

#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

#         if True in matches:
#             match_index = np.argmin(face_distances)
#             name = known_face_names[match_index]
#             print(f"Existing face recognized as: {name}")
#         else:
#             name = name_from_filename
#             print(f"New face detected. Assigned name: {name}")
#             add_new_face(face_encoding, name)

#         names.append(name)
#         face_info.append({"name": name, "box": {"top": top, "right": right, "bottom": bottom, "left": left}})
        
#         # Draw rectangle and name on the image
#         draw.rectangle(((left, top), (right, bottom)), outline="red", width=4)
#         draw.text((left + 6, bottom - 10), name, fill="red")

#     pil_image.save(output_image_path)
#     print(f"Processed image saved to: {output_image_path}")

#     # Prepare JSON output data
#     output_data = {
#         "image": output_image_path,
#         "names": names,
#         "faces": face_info
#     }

#     return output_data

# def main():
#     # Define input and output directories
#     input_dir = r'Images'
#     output_dir = r'Output\CNN'
#     os.makedirs(output_dir, exist_ok=True)
#     all_results = []  # List to store results for all images

#     # Process each image in the input directory
#     for filename in os.listdir(input_dir):
#         if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more formats if needed
#             input_image_path = os.path.join(input_dir, filename)
#             output_image_path = os.path.join(output_dir, filename)
            
#             result = process_image(input_image_path, output_image_path)
#             all_results.append(result)  # Append the result to the all_results list
#             print(result)  # Print the result for each image

#     # Save all results to a single JSON file
#     json_output_path = os.path.join(output_dir, 'all_results.json')
#     with open(json_output_path, 'w') as json_file:
#         json.dump(all_results, json_file, indent=4)
#     print(f"All results saved to: {json_output_path}")

# if __name__ == "__main__":
#     main()


import os
import json
from PIL import Image, ImageDraw
import numpy as np
import face_recognition

# Initialize known faces lists
known_face_encodings = []  # List to hold known face encodings
known_face_names = []      # List to hold names corresponding to known face encodings
unique_faces_dir = "Unique_Faces"  # Directory for storing unique face crops
os.makedirs(unique_faces_dir, exist_ok=True)

# Function to add a new face to the known faces database
def add_new_face(face_encoding, name, face_crop):
    thumbnail_path = os.path.join(unique_faces_dir, f"{name}.png")
    face_crop.save(thumbnail_path)  # Save the cropped face image
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    return thumbnail_path

# Function to process an image and detect/recognize faces
def process_image(input_image_path, output_image_path):
    """
    Processes the input image to detect and recognize faces,
    draws rectangles and names on recognized faces, and saves the output image.
    """
    if not os.path.exists(input_image_path):
        print(f"Input image file does not exist: {input_image_path}")
        return {"names": []}, []  # Return empty unique faces info as well

    image_file = Image.open(input_image_path)
    image = np.array(image_file)

    face_locations = face_recognition.face_locations(image, model="hog")
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_locations:
        print("No faces detected in the image.")
        return {"names": []}, []  # Return empty unique faces info if no faces

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    names = []
    face_info = []
    unique_faces_info = []  # To store unique face data for output JSON

    base_name = os.path.basename(input_image_path)
    name_from_filename = os.path.splitext(base_name)[0]

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if True in matches:
            match_index = np.argmin(face_distances)
            name = known_face_names[match_index]
            print(f"Existing face recognized as: {name}")
        else:
            name = name_from_filename
            print(f"New face detected. Assigned name: {name}")
            face_crop = pil_image.crop((left, top, right, bottom))  # Crop the detected face
            thumbnail_path = add_new_face(face_encoding, name, face_crop)
            unique_faces_info.append({
                "name": name,
                "thumbnail": thumbnail_path,
                "encoding": face_encoding.tolist()
            })

        names.append(name)
        face_info.append({"name": name, "box": {"top": top, "right": right, "bottom": bottom, "left": left}})

        # Draw rectangle and name on the image
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=4)
        draw.text((left + 6, bottom - 10), name, fill="red")

    pil_image.save(output_image_path)
    print(f"Processed image saved to: {output_image_path}")

    # Prepare JSON output data
    output_data = {
        "image": output_image_path,
        "names": names,
        "faces": face_info
    }

    return output_data, unique_faces_info

# Main function to process all images in the input directory
def main():
    # Define input and output directories
    input_dir = r'Images'
    output_dir = r'Output/CNN'
    os.makedirs(output_dir, exist_ok=True)

    # Check if input directory exists and contains image files
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No image files found in {input_dir}.")
        return

    all_results = []
    unique_faces_list = []

    # Process each image in the input directory
    for filename in image_files:
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)

        try:
            print(f"Processing image: {input_image_path}")
            result, unique_faces_info = process_image(input_image_path, output_image_path)
            all_results.append(result)
            unique_faces_list.extend(unique_faces_info)
            print(f"Finished processing: {input_image_path}")
        except Exception as e:
            print(f"Error processing {input_image_path}: {e}")
            continue

    # Save all recognition results to a single JSON file
    json_output_path = os.path.join(output_dir, 'all_results.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)
    print(f"All recognition results saved to: {json_output_path}")

    # Save unique faces data to another JSON file
    unique_faces_json_path = os.path.join(unique_faces_dir, 'unique_faces.json')
    with open(unique_faces_json_path, 'w') as json_file:
        json.dump(unique_faces_list, json_file, indent=4)
    print(f"Unique faces data saved to: {unique_faces_json_path}")

# Run the main function
if __name__ == "__main__":
    main()
