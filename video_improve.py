import cv2


def reduce_noise(image):
    if image is None:
        return "Image not loaded correctly"
    denoised_image = cv2.medianBlur(image, 3)
    return denoised_image


def draw_grid(image, rows, cols, color=(128, 128, 128), thickness=1):
    if image is None:
        return "Image not loaded correctly"
    height, width = image.shape[:2]
    cell_width = width // cols
    cell_height = height // rows

    # Рисуем вертикальные линии
    for i in range(1, cols):
        x = i * cell_width
        cv2.line(image, (x, 0), (x, height), color, thickness)

    # Рисуем горизонтальные линии
    for i in range(1, rows):
        y = i * cell_height
        cv2.line(image, (0, y), (width, y), color, thickness)

    return image

def draw_square(image, top_left, bottom_right, color=(0,255,0), thickness = 2):
    cv2.rectangle(image, top_left,bottom_right,color, thickness)
    mid_x = (top_left[0] + bottom_right[0]) // 2


    # Draw a dotted line in the middle
    cv2.line(image, (mid_x, top_left[1]), (mid_x, bottom_right[1]), (0,0,0), 1, lineType=cv2.LINE_AA)

    return image
top_left = (100,100)
bottom_right = (300,300)