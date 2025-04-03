import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch
import torchvision.models as models


def detection(image_path, yolo_model):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Не удалось загрузить изображение по пути: {image_path}')

    results = yolo_model(image, imgsz=640, iou=0.8, conf=0.4, verbose=True)
    if len(results[0].boxes) == 0:  # если ничего не нашли
        print('Объекты не обнаружены')
        return None, image

    # мы предполагаем, что родинка — единственный или первый найденный объект.
    # что делать со множественным обнаружением пока не ясно
    box = results[0].boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    return box, image


def crop_image(image, box):  # обрезка изображения по bounding_box
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    return cropped


def segmentation(cropped_image, seg_model, device):  # уже кропнутое изображение сегментируем
    transform = transforms.Compose([  # трансформируем как и обучали
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(input_tensor)['out']
        output = F.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
        preds = torch.sigmoid(output) > 0.5

    mask = preds.squeeze().cpu().numpy().astype(np.uint8)

    mask_resized = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
    return mask_resized


def dullrazor(image):  # https://github.com/BlueDokk/Dullrazor-algorithm/blob/main/dullrazor.py
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grayScale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), 0, borderType=cv2.BORDER_DEFAULT)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)
    '''cv2.imshow("Original image", image)
    cv2.imshow("Gray Scale image", grayScale)
    cv2.imshow("Blackhat", blackhat)
    cv2.imshow("Binary mask", mask)
    cv2.imshow("Clean image", dst)'''
    return dst


def e_shaver_hair_removal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 3)  # медианный фильтр для подавления шумов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray_blurred, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, kernel, iterations=1)
    inpainted = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
    return inpainted, hair_mask


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 1
    seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    seg_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    seg_model.to(device)

    image_path = input('Введите путь к изображению: ').strip()

    yolo_model = YOLO('C:\\Users\\kalin\\PycharmProjects\\pythonProject1\\runs\\detect\\train2\\weights\\best.pt')

    # Детекция родинки
    box, image = detection(image_path, yolo_model)
    if box is None:
        print('Родинка не обнаружена. Завершаем работу.')
        return

    # визуализация bounding_box для контроля
    image_with_box = image.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
    image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.title('Детекция: Bounding Box')
    plt.imshow(image_with_box_rgb)
    plt.axis('off')
    plt.show()

    # Обрезка изображения по найденному bounding box
    cropped = crop_image(image, box)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.title('Обрезанное изображение родинки')
    plt.imshow(cropped_rgb)
    plt.axis('off')
    plt.show()

    # Удаление волос (DullRazor)
    hair_removed = dullrazor(cropped)
    hair_removed_rgb = cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB)

    '''result, mask = e_shaver_hair_removal(cropped)
    hair_removed_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)'''
    plt.figure(figsize=(8, 8))
    plt.title('Изображение после удаления волос (DullRazor)')
    plt.imshow(hair_removed_rgb)
    plt.axis('off')
    plt.show()

    # Сегментация родинки
    seg_model.load_state_dict(torch.load('deeplabv3_15.03.25.pth', map_location=device))
    # mask = segmentation(cropped, seg_model, device)
    mask = segmentation(hair_removed_rgb, seg_model, device)

    # визуализация результата сегментации полупрозрачным синим цветом
    # overlay = cropped_rgb.copy()
    overlay = hair_removed_rgb.copy()
    overlay[mask == 1] = [0, 0, 255]
    alpha = 0.5
    # blended = cv2.addWeighted(cropped_rgb, 1 - alpha, overlay, alpha, 0)
    blended = cv2.addWeighted(hair_removed_rgb, 1 - alpha, overlay, alpha, 0)

    plt.figure(figsize=(8, 8))
    plt.title('Результат сегментации родинки')
    plt.imshow(blended)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
