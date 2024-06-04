from PIL import Image, ImageDraw

# Load the images
normal_img_path = "C:/Users/Eren/Desktop/DATASET/normal/NORMAL-48053-2.jpeg"
drusen_img_path = "C:/Users/Eren/Desktop/DATASET/drusen/DRUSEN-228939-9.jpeg"

normal_img = Image.open(normal_img_path)
drusen_img = Image.open(drusen_img_path)

# Create drawing contexts
normal_draw = ImageDraw.Draw(normal_img)
drusen_draw = ImageDraw.Draw(drusen_img)

# Define bounding boxes for normal and drusen areas
# Coordinates are in (x, y, x, y) format
normal_bbox = [(10, 10, 140, 140)]  # Placeholder values
drusen_bbox = [(10, 60, 150, 200)]  # Placeholder values

# Draw rectangles on the images
for bbox in normal_bbox:
    normal_draw.rectangle(bbox, outline="green", width=3)

for bbox in drusen_bbox:
    drusen_draw.rectangle(bbox, outline="red", width=3)

# Save the edited images
normal_img_output_path = "C:/Users/Eren/Desktop/NORMAL_annotated.jpeg"
drusen_img_output_path = "C:/Users/Eren/Desktop/DRUSEN_annotated.jpeg"

normal_img.save(normal_img_output_path)
drusen_img.save(drusen_img_output_path)

normal_img_output_path, drusen_img_output_path
