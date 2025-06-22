import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os # Import the os module for path operations

def analyze_astigmatism():
    """
    Analyzes an eye image to detect potential astigmatism by fitting an
    ellipse to the largest contour (assumed to be the cornea).
    Allows the user to select an image file via a dialog.
    """
    # Create a Tkinter root window (it won't be shown)
    root = tk.Tk()
    root.withdraw() # Hide the main window

    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename(
        title="Select an Eye Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )

    # Check if a file was selected
    if not image_path:
        print("No image selected. Exiting.")
        return

    print(f"Selected image: {os.path.basename(image_path)}") # Print only the filename for brevity

    # Load and preprocess image
    image = cv2.imread(image_path)

    # Error handling for image loading
    if image is None:
        print(f"Error: Could not load image from '{image_path}'. "
              "Please check if the file exists and is a valid image format.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Canny edge detection
    # Thresholds (30, 150) can be adjusted based on image quality
    edges = cv2.Canny(blurred, 30, 150)

    # Find contours in the edged image
    # RETR_EXTERNAL retrieves only the outermost contours
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Error handling if no contours are found
    if not contours:
        print("No contours found in the image. "
              "Ensure the image clearly shows the eye/cornea with good contrast.")
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image (No Contours Found)")
        plt.axis("off")
        plt.show()
        return

    # Find the largest contour based on area (assumed to be the cornea or main eye structure)
    largest_contour = max(contours, key=cv2.contourArea)

    # Check if the largest contour has enough points to fit an ellipse
    # An ellipse requires at least 5 points
    if len(largest_contour) < 5:
        print("Not enough points found in the largest contour to fit an ellipse. "
              "The contour might be too small, fragmented, or poorly detected.")
        plt.figure(figsize=(8, 6))
        plt.imshow(edges, cmap='gray')
        plt.title("Edges Detected (Insufficient Points for Ellipse Fit)")
        plt.axis("off")
        plt.show()
        return

    # Fit an ellipse to the largest contour
    ellipse = cv2.fitEllipse(largest_contour)
    (x, y), (major_axis, minor_axis), angle = ellipse

    # Create a copy of the original image to draw the ellipse on
    result_img = image.copy()
    cv2.ellipse(result_img, ellipse, (0, 255, 0), 2) # Draw ellipse in green with thickness 2

    # Calculate astigmatism index (axis ratio)
    # Handle cases where minor_axis might be zero to prevent division by zero error
    if min(major_axis, minor_axis) == 0:
        print("Warning: Minor axis is zero. Cannot calculate a meaningful axis ratio. "
              "This may indicate a highly elongated or degenerate ellipse fit.")
        axis_ratio = float('inf') # Represent as infinity
    else:
        axis_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

    # Display calculated metrics
    print(f"\n--- Analysis Results ---")
    print(f"Center of Ellipse: ({x:.2f}, {y:.2f})")
    print(f"Major Axis Length: {major_axis:.2f} pixels")
    print(f"Minor Axis Length: {minor_axis:.2f} pixels")
    print(f"Orientation Angle: {angle:.2f} degrees")
    print(f"Axis Ratio (Major/Minor): {axis_ratio:.2f}")

    # Interpretation of astigmatism based on axis ratio
    # Note: The threshold of 1.1 is a simplified example.
    # Actual clinical thresholds for astigmatism are more complex
    # and depend on diopter measurements, not just pixel ratios.
    if axis_ratio > 1.1:
        print("\nInterpretation: Possible Astigmatism Detected.")
        print("The cornea's shape appears significantly elliptical.")
    else:
        print("\nInterpretation: No Significant Astigmatism Indicated by Shape Ratio.")
        print("The cornea's shape appears relatively spherical or mildly elliptical.")

    print("\n------------------------")

    # Show the result image with the fitted ellipse
    plt.figure(figsize=(10, 8)) # Larger figure size for better viewing
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
    plt.title("Ellipse Fit of Cornea")
    plt.axis("off") # Hide axes for cleaner image display
    plt.show()

# Run the analysis when the script is executed
if __name__ == "__main__":
    analyze_astigmatism()
