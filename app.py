import cv2
import os
import ollama
import time


def analyze_image(image_path, object_str):
    """
    Analyze a single image to detect whether the target object is present.
    Args:
        image_path: Path to the image file
        object_str: Description of the target object to detect

    Returns:
        tuple: (Match or not, description text, confidence level)
    """
    prompt_str = f"""Please analyze the image and answer the following questions:

1. Is there a {object_str} in the image?
2. If yes, describe its appearance and location in the image in detail.
3. If no, describe what you see in the image instead.
4. On a scale of 1-10, how confident are you in your answer?

Please structure your response as follows:
Answer: [YES/NO]
Description: [Your detailed description]
Confidence: [1-10]"""

    try:
        # Use the Llama model to analyze the image
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[
                {
                    'role': 'user',
                    'content': prompt_str,
                    'images': [image_path]
                }
            ]
        )

        print(f"Waiting for the model to analyze...")
        time.sleep(1)  # Brief delay to ensure the response is complete

        # Get and print the raw response
        response_text = response['message']['content']
        print(f"Raw response: {response_text}")

        # Process the response text, removing Markdown formatting
        response_text = response_text.replace('**', '')
        response_lines = response_text.strip().split('\n')

        # Extract key information from the response
        answer = None
        description = None
        confidence = 10  # Default confidence level is 10, since the model does not explicitly return confidence

        # Parse the response content line by line
        for line in response_lines:
            line = line.strip()
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip().upper()
            # Match Description, Reasoning, and Alternative Description
            elif any(line.lower().startswith(prefix) for prefix in
                     ['description:', 'reasoning:', 'alternative description:']):
                description = line.split(':', 1)[1].strip()
            elif line.lower().startswith('confidence:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 10  # Use default value if confidence cannot be parsed

        # Check if necessary information was obtained
        if answer is None or description is None:
            raise ValueError("Response format is incomplete")

        print(f"Parsed result - Answer: {answer}, Description: {description}, Confidence: {confidence}")

        # Return the analysis result
        return answer == "YES" and confidence >= 7, description, confidence
    except Exception as e:
        print(f"Error during image analysis: {e}")
        import traceback
        print(traceback.format_exc())
        return False, "Error occurred", 0


def preprocess_image(image_path):
    """
    Image preprocessing function to enhance image quality.
    Args:
        image_path: Path to the image file
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert the color space and enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Save the processed image
    cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])


def extract_and_analyze_frames(video_path, output_folder, object_str):
    """
    Extract frames from a video and analyze them to check for the presence of the target object.
    Args:
        video_path: Path to the video file
        output_folder: Folder to save frame images
        object_str: Description of the target object to detect

    Returns:
        int or None: The time point (in seconds) when the target was found, or None if not found
    """
    # Create the output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None

    # Get the video FPS
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    consecutive_matches = 0
    match_threshold = 1  # Consecutive match threshold
    cool_down_time = 2  # Cooldown time (seconds) after analyzing each frame

    print(f"Starting video analysis, FPS: {fps}")

    try:
        while True:
            # Read a frame from the video
            success, frame = video.read()
            if not success:
                break

            # Process one frame per second
            if frame_count % fps == 0:
                print(f"\nProcessing frame at {frame_count // fps} seconds")

                # Save the current frame
                output_filename = os.path.join(output_folder, f"frame_{frame_count // fps}.jpg")
                output_filename = os.path.abspath(output_filename)

                cv2.imwrite(output_filename, frame)
                print(f"Saved frame to: {output_filename}")

                # Preprocess the image
                preprocess_image(output_filename)
                print("Completed image preprocessing")

                print("Starting image analysis...")
                print(f"Using image path: {output_filename}")

                # Check if the file exists
                if not os.path.exists(output_filename):
                    print(f"Warning: File does not exist: {output_filename}")
                    continue

                # Analyze the image
                is_match, description, confidence = analyze_image(output_filename, object_str)
                print(f"Analysis complete - Match: {is_match}, Confidence: {confidence}")
                print(f"Description: {description}")

                # Handle the match result
                if is_match:
                    consecutive_matches += 1
                    print(f"Potential match - Time: {frame_count // fps} seconds")
                    print(f"Description: {description}")
                    print(f"Confidence: {confidence}")

                    # If the number of consecutive matches reaches the threshold, return the result and exit
                    if consecutive_matches >= match_threshold:
                        match_time = frame_count // fps - match_threshold + 1
                        print(f"Found consecutive matches! Time: From {match_time} to {frame_count // fps} seconds")
                        video.release()  # Release the video resource
                        return match_time  # Return the result
                else:
                    consecutive_matches = 0

                # Cooldown time after analyzing a frame
                print(f"Waiting {cool_down_time} seconds to cool down the GPU...")
                time.sleep(cool_down_time)

            frame_count += 1

    finally:
        # Ensure the video resource is released
        video.release()

    print(f"No matching images found. A total of {frame_count // fps} frames were analyzed.")
    return None


# Main entry point
if __name__ == "__main__":
    # Set parameters
    video_path = "./a.mp4"
    output_folder = "output_frames"
    object_to_find = "A red car driving on the road"

    print("Starting the video analysis program...")
    # Run the analysis
    result = extract_and_analyze_frames(video_path, output_folder, object_to_find)

    # Output the result
    if result is not None:
        print(f"The target object was found at {result} seconds in the video.")
    else:
        print("The target object was not found in the video.")
