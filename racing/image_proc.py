import cv2
import numpy as np

# Step 1: Load the Image
image_path = 'YASS.png'  # Replace with your image file path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image from path '{image_path}'")
    exit(1)

# Step 2: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Binary Inverse Thresholding
# This should highlight the black track as white (255) and everything else as black (0)
_, binary_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# Step 4: Use Morphological Operations to Clean Up (Optional)
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 5: Create Mask to Remove Image Frame
height, width = cleaned.shape
mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(mask, (1, 1), (width-2, height-2), 255, -1)  # Leave a 1-pixel border

# Apply mask to the cleaned image
masked_image = cv2.bitwise_and(cleaned, mask)

# Step 6: Find Contours (Before Filtering)
contours, _ = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Check how many contours are found
print(f"Contours found: {len(contours)}")

# Step 7: Visualize Contours for Debugging
# Draw all contours on the original image to see if we detected anything
debug_image = image.copy()
cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)

# Display the debug image to see the contours
cv2.imshow("Contours Debug", debug_image)
cv2.imshow("Masked Image", masked_image)
cv2.imshow("Threshold Image", binary_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 8: Filter Contours (If Needed)
# For testing, remove area filtering and see if the contours are useful
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # Increase the threshold as needed

# If filtering results in no contours, print a message and proceed to the next step
if len(filtered_contours) == 0:
    print("No contours after filtering based on area.")
else:
    print(f"Found {len(filtered_contours)} contours after filtering.")

# Step 9: Extract Coordinates and Save to a File (Optional)
output_file = "track_coordinates.txt"
with open(output_file, "w") as file:
    for contour in filtered_contours:
        # Extract coordinates from each filtered contour
        for point in contour:
            x, y = point[0]
            file.write(f"{x}, {y}\n")

    print(f"Saved track coordinates to '{output_file}'")

# Optional: Visualize the final filtered contours
final_image = image.copy()
cv2.drawContours(final_image, filtered_contours, -1, (0, 255, 0), 2)
cv2.imshow("Final Contours", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# File paths
input_file = "track_coordinates.txt"  # Original file with coordinates
output_file = "resized_track_coordinates.txt"  # Output file for resized coordinates

# Original and target sizes
original_width = 1200
original_height = 800
target_width = 1200
target_height = 800

# Scaling factor for y-coordinate (since width is unchanged)
y_scale = target_height / original_height  # 800 / 1000 = 0.8

# Open the input file to read the coordinates
with open(input_file, "r") as infile:
    lines = infile.readlines()

# Open the output file to write the resized coordinates
with open(output_file, "w") as outfile:
    for line in lines:
        # Read each line, which contains x, y coordinates
        x, y = map(int, line.strip().split(", "))

        # Resize the coordinates
        resized_x = x  # x-coordinate stays the same
        resized_y = int(y * y_scale)  # Scale y-coordinate

        # Write the resized coordinates to the new file
        outfile.write(f"{resized_x}, {resized_y}\n")

print(f"Resized coordinates saved to '{output_file}'")

