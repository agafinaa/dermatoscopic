import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import transformers
from torch import nn
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch
import torchvision.models as models
from transformers import ViTForImageClassification
from torchvision.transforms import InterpolationMode
from skimage.morphology import remove_small_objects
from scipy.signal import wiener
from realesrgan import RealESRGANer
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet
import onnxruntime as ort



class ResNetFeatMask(nn.Module):
    def __init__(self, num_classes=2, pretrained_backbone=True):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained_backbone)
        self.features = nn.Sequential(*(list(backbone.children())[:-2]))
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, seg_mask=None):
        feat = self.features(x)
        if seg_mask is not None:
            mask_small = F.interpolate(seg_mask, size=feat.shape[2:], mode='nearest')
            feat = feat * mask_small
        out = self.avgpool(feat)
        out = torch.flatten(out, 1)
        return self.fc(out)


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


def convert_to_grayscale(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def zero_crossing(lap, threshold):
    sign = np.sign(lap)
    sign_right = np.roll(sign, -1, axis=1)
    sign_down = np.roll(sign, -1, axis=0)
    diff_right = np.abs(lap - np.roll(lap, -1, axis=1))
    diff_down = np.abs(lap - np.roll(lap, -1, axis=0))
    zc_candidate = (((sign * sign_right) < 0) & (diff_right > threshold)) | \
                   (((sign * sign_down) < 0) & (diff_down > threshold))
    zc = (zc_candidate.astype(np.uint8)) * 255
    zc[0, :] = 0
    zc[-1, :] = 0
    zc[:, 0] = 0
    zc[:, -1] = 0
    return zc


def matlab_like_log_edge(gray, sigma=2.0, relative_threshold=0.03):
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(blurred, ddepth=cv2.CV_32F, ksize=3)
    abs_max = np.max(np.abs(lap))
    abs_threshold = relative_threshold * abs_max
    edge_mask = zero_crossing(lap, threshold=abs_threshold)
    return edge_mask


def subtract_images(gray, lap_gray):
    return cv2.subtract(gray, lap_gray)


def obtain_binary_mask_log(image, ksize=5, threshold_val=30):
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    lap_abs = np.uint8(np.absolute(lap))
    _, binary_mask = cv2.threshold(lap_abs, threshold_val, 255, cv2.THRESH_BINARY)
    return binary_mask


def obtain_binary_mask_sobel(image, threshold_val=100):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.absolute(sobel))
    _, binary_mask = cv2.threshold(sobel, threshold_val, 255, cv2.THRESH_BINARY)
    return binary_mask


def add_binary_images(mask1, mask2):
    return cv2.bitwise_or(mask1, mask2)


def wiener_filter(image):
    filtered = wiener(image)
    #filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    filtered = np.clip(filtered, 0.0, 255.0).astype(np.uint8)
    return filtered


def convert_to_binary(image, threshold_val=128):
    _, binary = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)
    return binary


def morphological_closing(mask, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def morphological_dilation(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def morphological_erosion(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=1)


def channel_hair_pixel_replacement_interpolation(channel, mask):
    pseudo_bgr = cv2.merge([channel, channel, channel])
    repaired_bgr = cv2.inpaint(pseudo_bgr, mask, 3, cv2.INPAINT_TELEA)
    repaired_channel = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2GRAY)
    return repaired_channel


def replace_hair_pixels_in_all_channels(img_bgr, mask):
    b_channel, g_channel, r_channel = cv2.split(img_bgr)
    repaired_r = channel_hair_pixel_replacement_interpolation(r_channel, mask)
    repaired_g = channel_hair_pixel_replacement_interpolation(g_channel, mask)
    repaired_b = channel_hair_pixel_replacement_interpolation(b_channel, mask)
    return cv2.merge([repaired_b, repaired_g, repaired_r])


def matlab_like_laplacian_filtering(gray, alpha=0.2):
    kernel = np.array([[alpha, 1 - alpha, alpha],
                       [1 - alpha, -4 * (1 - alpha), 1 - alpha],
                       [alpha, 1 - alpha, alpha]], dtype=np.float32)
    filtered = cv2.filter2D(gray, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    return filtered


def nonmax_suppression(mag, angle):
    rows, cols = mag.shape
    nms = np.zeros_like(mag, dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            a = angle[i, j]
            m = mag[i, j]
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                neighbor1 = mag[i, j - 1]
                neighbor2 = mag[i, j + 1]
            elif 22.5 <= a < 67.5:
                neighbor1 = mag[i - 1, j + 1]
                neighbor2 = mag[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                neighbor1 = mag[i - 1, j]
                neighbor2 = mag[i + 1, j]
            elif 112.5 <= a < 157.5:
                neighbor1 = mag[i - 1, j - 1]
                neighbor2 = mag[i + 1, j + 1]
            else:
                neighbor1 = 0
                neighbor2 = 0

            if m >= neighbor1 and m >= neighbor2:
                nms[i, j] = m
            else:
                nms[i, j] = 0
    return nms


def matlab_like_sobel_edge(gray, relative_threshold=0.1):
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    angle = np.arctan2(Gy, Gx) * (180.0 / np.pi)
    angle[angle < 0] += 180
    nms = nonmax_suppression(mag, angle)
    max_nms = np.max(nms)
    thresh = relative_threshold * max_nms
    edge_mask = np.zeros_like(nms, dtype=np.uint8)
    edge_mask[nms >= thresh] = 255
    return edge_mask


def matlab_like_wiener_noise_reduction(image, kernel_size=(5, 5)):
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    local_mean = cv2.blur(image, kernel_size)
    local_mean_sq = cv2.blur(image * image, kernel_size)
    local_variance = local_mean_sq - local_mean * local_mean
    noise_variance = np.mean(local_variance)
    epsilon = 1e-8
    result = local_mean + (np.maximum(local_variance - noise_variance, 0) / (local_variance + epsilon)) * (image - local_mean)
    filtered = np.clip(result * 255, 0, 255).astype(np.uint8)
    return filtered


def run_lls(input_path):
    img_bgr = input_path
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    '''cv2.namedWindow("Convert to Grayscale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Convert to Grayscale", 300, 300)
    cv2.imshow("Convert to Grayscale", gray)
    cv2.waitKey(0)'''

    #lap_gray = matlab_like_log_edge(gray, sigma=2.0, relative_threshold=0.03)
    lap_gray = matlab_like_laplacian_filtering(gray, alpha=0.2)
    '''plt.figure(figsize=(8, 6))
    plt.imshow(lap_gray, cmap='gray')
    plt.title("MATLAB-like Laplacian Filtering")
    plt.axis("off")
    plt.show()'''

    subtracted_image = subtract_images(gray, lap_gray)
    #cv2.imshow("Subtracted Image", subtracted_image)
    '''plt.figure(figsize=(8, 6))
    plt.imshow(subtracted_image, cmap='gray')
    plt.title("Subtract images")
    plt.axis("off")
    plt.show()'''

    # binary_mask_log = obtain_binary_mask_log(subtracted_image, ksize=5, threshold_val=30)
    binary_mask_log = matlab_like_log_edge(subtracted_image, sigma=3.0, relative_threshold=0.09)
    # binary_mask_sobel = obtain_binary_mask_sobel(subtracted_image, threshold_val=50)
    binary_mask_sobel = matlab_like_sobel_edge(subtracted_image, relative_threshold=0.2)
    '''plt.figure(figsize=(8, 6))
    plt.imshow(binary_mask_log, cmap='gray')
    plt.title("Бинарная маска Log")
    plt.axis("off")
    plt.show()
    plt.figure(figsize=(8, 6))
    plt.imshow(binary_mask_sobel, cmap='gray')
    plt.title("Бинарная маска Sobel")
    plt.axis("off")
    plt.show()'''

    added_mask = add_binary_images(binary_mask_log, binary_mask_sobel)
    '''plt.figure(figsize=(8, 6))
    plt.imshow(added_mask, cmap='gray')
    plt.title("Add binary images")
    plt.axis("off")
    plt.show()'''

    #wiener_result = wiener_filter(added_mask)
    wiener_result = matlab_like_wiener_noise_reduction(added_mask, kernel_size=(5, 5))
    '''plt.figure(figsize=(8, 6))
    plt.imshow(wiener_result, cmap='gray')
    plt.title("Wiener noise reduction filter")
    plt.axis("off")
    plt.show()'''

    binary_after_wiener = convert_to_binary(wiener_result, threshold_val=128)
    '''plt.figure(figsize=(8, 6))
    plt.imshow(binary_after_wiener, cmap='gray')
    plt.title("Binary After Wiener")
    plt.axis("off")
    plt.show()'''

    morph_mask = morphological_closing(binary_after_wiener, kernel_size=7)
    morph_mask = morphological_dilation(morph_mask, kernel_size=3)
    morph_mask = morphological_erosion(morph_mask, kernel_size=3)
    '''plt.figure(figsize=(8, 6))
    plt.imshow(morph_mask, cmap='gray')
    plt.title("Морфологические операции")
    plt.axis("off")
    plt.show()'''

    final_img = replace_hair_pixels_in_all_channels(img_bgr, morph_mask)
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(final_img_rgb, cmap='gray')
    plt.title("Final Processed Image")
    plt.axis("off")
    plt.show()


def enhance_image(input_path, output_path, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGANer(
        scale=scale,
        model_path='C:\\Users\\kalin\\PycharmProjects\\pythonProject1\\Real-ESRGAN\\weights\\RealESRGAN_x4plus.pth',  # путь к файлу предобученной модели
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути {input_path}")
    output_img, _ = model.enhance(img, outscale=scale)

    cv2.imwrite(output_path, output_img)
    print(f"Изображение успешно сохранено как {output_path}")


def preprocess_image(img, scale):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_rgb, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor


def postprocess_output(output):
    output = np.squeeze(output, axis=0)  # (3, H, W)
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def run_onnx_realesrgan(input_image_path, output_image_path, model_path, outscale=4):
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Изображение не найдено по пути: {input_image_path}")

    # Предобработка
    input_tensor = preprocess_image(img, scale=outscale)
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: input_tensor})
    output_tensor = ort_outputs[0]

    # Постобработка и сохранение результата
    output_img = postprocess_output(output_tensor)
    cv2.imwrite(output_image_path, output_img)
    print(f"Улучшенное изображение сохранено по пути: {output_image_path}")


def main():
    '''input_path = "волосы2.jpg"  # Укажите ваш путь к входному изображению
    output_path = "волосы2-new.jpg"  # Путь для сохранения результата
    onnx_model_path = "RealESRGAN_x4plus.onnx"  # Путь к ONNX‑модели (например, преобразованной версии Real-ESRGAN x4plus)

    run_onnx_realesrgan(input_path, output_path, onnx_model_path)'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 1
    seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    seg_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    seg_model.to(device)

    image_path = input('Введите путь к изображению: ').strip()

    # enhance_image(image_path, 'photo_2025-04-01_16-19-11-1.jpg', scale=4)

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
    '''plt.figure(figsize=(8, 8))
    plt.title('Детекция: Bounding Box')
    plt.imshow(image_with_box_rgb)
    plt.axis('off')
    plt.show()'''

    # Обрезка изображения по найденному bounding box
    cropped = crop_image(image, box)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    '''plt.figure(figsize=(8, 8))
    plt.title('Обрезанное изображение родинки')
    plt.imshow(cropped_rgb)
    plt.axis('off')
    plt.show()'''

    # run_lls(cropped)

    # Удаление волос (DullRazor)
    hair_removed = dullrazor(cropped)
    hair_removed_rgb = cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB)

    '''result, mask = e_shaver_hair_removal(cropped)
    hair_removed_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)'''
    '''plt.figure(figsize=(8, 8))
    plt.title('Изображение после удаления волос (DullRazor)')
    plt.imshow(hair_removed_rgb)
    plt.axis('off')
    plt.show()'''

    # Сегментация родинки
    seg_model.load_state_dict(torch.load('best_model_deeplabv3_26.04.25.pth', map_location=device))
    # mask = segmentation(cropped, seg_model, device)
    mask = segmentation(hair_removed_rgb, seg_model, device)

    '''plt.figure(figsize=(8, 8))
    plt.title('Маска')
    plt.imshow(mask)
    plt.axis('off')
    plt.show()'''

    # визуализация результата сегментации полупрозрачным синим цветом
    # overlay = cropped_rgb.copy()
    '''overlay = hair_removed_rgb.copy()
    overlay[mask == 1] = [0, 0, 255]
    alpha = 0.5
    # blended = cv2.addWeighted(cropped_rgb, 1 - alpha, overlay, alpha, 0)
    blended = cv2.addWeighted(hair_removed_rgb, 1 - alpha, overlay, alpha, 0)

    plt.figure(figsize=(8, 8))
    plt.title('Результат сегментации родинки')
    plt.imshow(blended)
    plt.axis('off')
    plt.show()'''

    # Бинарная классификация
    class_image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    class_mask_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t > 0.5).float())])

    model = ResNetFeatMask(num_classes=2, pretrained_backbone=True).to(device)
    model.load_state_dict(torch.load('C:\\Users\\kalin\\PycharmProjects\\pythonProject1\\best_f1_resnet_3.05.25.pth', map_location=device))
    model.to(device)

    pil_img = Image.fromarray(hair_removed_rgb)
    img_tensor = class_image_transform(pil_img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    mask_uint8 = (mask.squeeze() * 255).astype('uint8')
    pil_mask = Image.fromarray(mask_uint8, mode='L')
    mask_tensor = class_mask_transform(pil_mask)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)

    # 3) Инференс
    '''model.eval()
    with torch.no_grad():
        outputs = model(img_tensor, seg_mask=mask_tensor)
        pred = outputs.argmax(dim=1).item()

    label_map = {0: 'доброкачественная', 1: 'злокачественная'}
    print(f'Предсказание: {label_map[pred]}')'''

    with torch.no_grad():
        outputs = model(img_tensor, seg_mask=mask_tensor)
        probs = torch.softmax(outputs, dim=1)
        prob_malign = probs[0, 1].item()
        prob_benign = probs[0, 0].item()

    threshold = 0.3  # на 0.3 лучшее соотношение общих ошибок к раковым, базово поставить 0.5, но тогда меньше рака поймает
    pred = 1 if prob_malign >= threshold else 0

    label_map = {0: 'доброкачественное', 1: 'злокачественное'}
    # print(f"Вероятности — доброкачественная: {prob_benign:.3f}, злокачественная: {prob_malign:.3f}")
    # print(f"При пороге {threshold} предсказано: {label_map[pred]}")
    print(f"На предоставленном изображении: {label_map[pred]} новообразование")


if __name__ == '__main__':
    main()
