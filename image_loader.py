#!/usr/bin/env python3
import io
import os
import sys
import asyncio
import httpx
import logging
import mimetypes
from io import BytesIO
from datetime import datetime
from PIL import Image as PILImage
from urllib.parse import urlparse
from mcp.server.fastmcp import FastMCP, Image, Context
from ultralytics import YOLO
import numpy as np
import cv2
import time
from datetime import datetime

TEMP_DIR = "./Temp"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

log_filename = os.path.join(DATA_DIR, datetime.now().strftime("%d-%m-%y.log"))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(formatter)

logger = logging.getLogger("image-mcp")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

# Create a FastMCP server instance
mcp = FastMCP("single-image-service")

# Preload YOLO model to avoid runtime download delays
try:
    YOLO_MODEL = YOLO("yolov8n.pt")
    logger.debug("Preloaded YOLOv8n model")
except Exception as e:
    logger.error(f"Failed to preload YOLOv8 model: {str(e)}")
    YOLO_MODEL = None

try:
    YOLO_MODEL_SEG = YOLO("yolo11n-seg.pt")
    logger.debug("Preloaded YOLOv11n-seg model")
except Exception as e:
    logger.error(f"Failed to preload YOLOv11n-seg model: {str(e)}")
    YOLO_MODEL_SEG = None


async def process_image_data(source: str, data: bytes | None = None, ctx: Context | None = None) -> Image | None:
    """Process image data from bytes or a file path, returning an MCP Image object."""
    try:
        # Determine content type
        content_type, _ = mimetypes.guess_type(source)
        content_type = content_type.split('/')[-1] if content_type and content_type.startswith('image/') else 'jpeg'
        logger.debug(f"Processing image from {source} with content type: {content_type}")

        # Load image data if not provided
        if data is None:
            with open(source, "rb") as f:
                data = f.read()
            logger.debug(f"Read local image from {source} with {len(data)} bytes")

        # Log original dimensions
        try:
            with PILImage.open(BytesIO(data)) as img:
                width, height = img.size
                logger.debug(f"Original image dimensions from {source}: {width}x{height}, format: {img.format}, mode: {img.mode}")
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                    width, height = img.size
                    logger.debug(f"Converted to RGB, new dimensions: {width}x{height}")
                pil_img = img
        except Exception as e:
            logger.debug(f"Could not determine dimensions for {source}: {e}")
            pil_img = None

        # If image is small enough, return as-is
        if len(data) <= 1048576:
            return Image(data=data, format=content_type)

        # Compress large images
        if pil_img is None:
            with PILImage.open(BytesIO(data)) as img:
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                width, height = img.size
                pil_img = img

        quality = 85
        while True:
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='JPEG', quality=quality)
            if len(img_byte_arr.getvalue()) <= 1048576:
                logger.debug(f"Compressed image from {source} to {len(img_byte_arr.getvalue())} bytes (quality={quality})")
                return Image(data=img_byte_arr.getvalue(), format='jpeg')
            
            if quality > 30:
                quality -= 10
                logger.debug(f"Reducing quality to {quality} for {source}")
            else:
                width = int(width * 0.8)
                height = int(height * 0.8)
                if width < 200 or height < 200:
                    error_msg = f"Unable to compress image from {source} to acceptable size"
                    logger.error(error_msg)
                    if ctx: ctx.error(error_msg)
                    return None
                logger.debug(f"Resizing image from {source} to {width}x{height}")
                pil_img = pil_img.resize((width, height), PILImage.LANCZOS)
                quality = 85

    except Exception as e:
        error_msg = f"Error processing image from {source}: {str(e)}"
        logger.error(error_msg)
        if ctx: ctx.error(error_msg)
        return None

async def fetch_single_image(url: str, client: httpx.AsyncClient, ctx: Context) -> Image | None:
    """Fetch and process a single image from a URL."""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme in ['http', 'https'], parsed.netloc]):
            ctx.error(f"Invalid URL: {url}")
            return None

        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            ctx.error(f"Not an image (got {content_type})")
            return None

        logger.debug(f"Fetched image from {url} with {len(response.content)} bytes")
        return await process_image_data(url, response.content, ctx)

    except httpx.HTTPError as e:
        ctx.error(f"HTTP error: {str(e)}")
        return None
    except Exception as e:
        ctx.error(f"Unexpected error: {str(e)}")
        return None

def is_url(path_or_url: str) -> bool:
    """Determine if the given string is a URL or a local file path."""
    parsed = urlparse(path_or_url)
    return bool(parsed.scheme and parsed.netloc)

@mcp.tool()
async def fetch_image(image_source: str, ctx: Context) -> Image | None:
    """
    Fetch and process a single image from a URL or local file path, returning it in a format suitable for LLMs.

    Args:
        image_source: A URL (http:// or https://) or local file path pointing to an image.

    Returns:
        An Image object or None if processing fails.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Processing image source: {image_source}")

        if not image_source:
            ctx.error("No image source provided")
            return None

        if is_url(image_source):
            async with httpx.AsyncClient() as client:
                logger.debug(f"found a URL as image source ,URL:{image_source}")
                result = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                return None
            result = await process_image_data(image_source, ctx=ctx)

        elapsed = asyncio.get_event_loop().time() - start_time
        status = "Success" if result is not None else "Failed"
        logger.debug(f"Processed image from {image_source} in {elapsed:.2f} seconds. Status: {status}")
        return result

    except Exception as e:
        ctx.error(f"Failed to process image: {str(e)}")
        logger.error(f"Error in fetch_image: {str(e)}")
        return None

@mcp.tool()
async def get_image_dimensions_and_save(image_source: str, ctx: Context) -> dict | None:
    """
    Fetch an image from a URL or local file path, retrieve its dimensions, save it to the output directory,
    and return a dictionary with the dimensions and saved file path.

    This tool processes an image from either a URL or a local file path, determines its dimensions using PIL,
    saves the processed image to the 'output' directory with a timestamped filename, and returns a dictionary
    containing the width, height, and the path where the image was saved.

    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to an image.

    Returns:
        dict | None: A dictionary containing the image dimensions ('width', 'height') and the saved file path
                     ('saved_path'), or None if processing fails.

    Example:
        >>> result = await get_image_dimensions_and_save("https://example.com/image.jpg", ctx)
        >>> print(result)
        {'width': 800, 'height': 600, 'saved_path': './output/image_2025-07-06_211823.jpg'}
    """
    # Notes:
    #     - The image is saved in JPEG format to the 'output' directory with a timestamped filename.
    #     - If the image is too large (>1MB), it is compressed using the same logic as `process_image_data`.
    #     - The tool handles both local files and remote images via HTTP/HTTPS.
    #     - Errors during processing (e.g., invalid URL, file not found, or processing failure) are logged
    #       and reported via the context object, returning None.
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Processing image for dimensions and save: {image_source}")

        if not image_source:
            ctx.error("No image source provided")
            return None

        # Fetch or process the image
        if is_url(image_source):
            async with httpx.AsyncClient() as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            return None

        # Get dimensions
        try:
            with PILImage.open(BytesIO(image.data)) as img:
                width, height = img.size
                logger.debug(f"Image dimensions from {image_source}: {width}x{height}")
        except Exception as e:
            ctx.error(f"Could not determine dimensions for {image_source}: {str(e)}")
            logger.error(f"Error getting dimensions: {str(e)}")
            return None

        # Save the image to output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(save_path, "wb") as f:
                f.write(image.data)
            logger.debug(f"Saved image to {save_path}")
        except Exception as e:
            ctx.error(f"Failed to save image to {save_path}: {str(e)}")
            logger.error(f"Error saving image: {str(e)}")
            return None

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"Processed and saved image from {image_source} in {elapsed:.2f} seconds")
        return {"width": width, "height": height, "saved_path": save_path}

    except Exception as e:
        ctx.error(f"Failed to process image dimensions and save: {str(e)}")
        logger.error(f"Error in get_image_dimensions_and_save: {str(e)}")
        return None
    

# Add this constant near other directory constants (e.g., after OUTPUT_DIR definition)
SUPPORTED_FORMATS = {'jpeg', 'png', 'gif'}

@mcp.tool()
async def convert_image_format(image_source: str, target_format: str, ctx: Context, save: bool = True) -> Image | None:
    """
    Convert an image from a URL or local file path to a specified format and optionally save it to the output directory.

    This tool processes an image from a URL or local file, converts it to the specified format (e.g., 'jpeg', 'png', 'gif'),
    and returns an Image object. If requested, the converted image is saved to the 'output' directory with a timestamped
    filename. It reuses existing image processing logic for consistency and applies compression if needed.

    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to an image.
        target_format (str): The desired output format (e.g., 'jpeg', 'png', 'gif').
        save (bool): Whether to save the converted image to the output directory (default: True).

    Returns:
        Image | None: An Image object with the converted image data, or None if processing fails.

    Example:
        >>> result = await convert_image_format("https://example.com/image.png", "jpeg", save=True, ctx)
        >>> print(result.format)
        'jpeg'
    """ 

    # Notes:
    #     - Supported formats are JPEG, PNG, and GIF.
    #     - Images are compressed to stay under 1MB, consistent with existing processing logic.
    #     - The saved image filename includes a timestamp to avoid overwrites.
    #     - Errors (e.g., unsupported format, invalid source) are logged and reported via the context.
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Converting image from {image_source} to {target_format}")

        # Validate target format
        target_format = target_format.lower()
        if target_format not in SUPPORTED_FORMATS:
            ctx.error(f"Unsupported target format: {target_format}. Supported formats: {', '.join(SUPPORTED_FORMATS)}")
            return None

        if not image_source:
            ctx.error("No image source provided")
            return None

        # Fetch or process the image
        if is_url(image_source):
            async with httpx.AsyncClient() as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            return None

        # Convert to target format
        try:
            with PILImage.open(BytesIO(image.data)) as img:
                if img.mode in ('RGBA', 'P') and target_format != 'png':
                    img = img.convert('RGB')
                    logger.debug(f"Converted image mode to RGB for {target_format} output")

                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format=target_format.upper(), quality=85 if target_format == 'jpeg' else None)
                converted_data = img_byte_arr.getvalue()
                logger.debug(f"Converted image to {target_format}, size: {len(converted_data)} bytes")
        except Exception as e:
            ctx.error(f"Failed to convert image to {target_format}: {str(e)}")
            logger.error(f"Conversion error: {str(e)}")
            return None

        # Save the converted image if requested
        if save:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"image_{timestamp}.{target_format}"
            save_path = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(save_path, "wb") as f:
                    f.write(converted_data)
                logger.debug(f"Saved converted image to {save_path}")
            except Exception as e:
                ctx.error(f"Failed to save converted image to {save_path}: {str(e)}")
                logger.error(f"Save error: {str(e)}")

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"Converted image from {image_source} to {target_format} in {elapsed:.2f} seconds")
        return Image(data=converted_data, format=target_format)

    except Exception as e:
        ctx.error(f"Failed to convert image: {str(e)}")
        logger.error(f"Error in convert_image_format: {str(e)}")
        return None
    

# Add this near the top with other imports

@mcp.tool()
async def detect_objects_yolo(image_source: str, ctx: Context,save: bool = False) -> dict | None:
    """
    Perform object detection on an image using YOLOv8 and return detailed results as a dictionary.

    This tool processes an image from a URL or local file path, performs object detection using the YOLOv8 model,
    and returns a dictionary containing the detected objects' labels, confidence scores, bounding box coordinates,
    and an explanation of the results. If requested, the input image with bounding boxes drawn is saved to the
    'output' directory with a timestamped filename. It reuses existing image processing logic for loading and handling.
    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to an image.
        save (bool): Whether to save the input image to the output directory (default: True).

    Returns:
        dict | None: A dictionary containing:
            - 'detections': A list of dictionaries, each with:
                - 'label': The class name of the detected object (e.g., 'person', 'car').
                - 'confidence': The confidence score (0.0 to 1.0) of the detection.
                - 'bbox': A dictionary with 'x_min', 'y_min', 'x_max', 'y_max' coordinates of the bounding box.
            - 'explanation': A dictionary explaining the results, including:
                - 'total_detections': Number of detected objects.
                - 'labels_detected': Unique class labels found.
                - 'image_source': The input image source.
                - 'saved_path': Path where the image was saved (if save=True).
                - 'model_used': The YOLO model used (e.g., 'yolov8n').
                - 'bbox_format': Description of the bounding box coordinate format.
                - 'confidence_description': Explanation of confidence scores.
        Returns None if processing or detection fails.

    Example:
        >>> result = await detect_objects_yolo("https://example.com/image.jpg", save=True, ctx)
        >>> print(result)
        {
            'detections': [
                {'label': 'person', 'confidence': 0.85, 'bbox': {'x_min': 100, 'y_min': 150, 'x_max': 200, 'y_max': 300}},
                {'label': 'car', 'confidence': 0.92, 'bbox': {'x_min': 250, 'y_min': 200, 'x_max': 400, 'y_max': 350}}
            ],
            'explanation': {
                'total_detections': 2,
                'labels_detected': ['person', 'car'],
                'image_source': 'https://example.com/image.jpg',
                'saved_path': './output/image_2025-07-07_113423.jpg',
                'model_used': 'yolov8n',
                'bbox_format': 'Bounding box coordinates (x_min, y_min, x_max, y_max) represent the top-left and bottom-right corners of the detected object in the image.',
                'confidence_description': 'Confidence scores (0.0 to 1.0) indicate the model’s certainty that the detection is correct; higher is more confident.'
            }
        }
    """
    # Notes:
    #     - Requires the `ultralytics` library (`pip install ultralytics`) for YOLOv8.
    #     - Uses the lightweight 'yolov8n' model for efficiency.
    #     - Images are processed using existing compression logic to ensure compatibility.
    #     - The saved image is the original input image, not the annotated output, to maintain consistency with other tools.
    #     - Errors (e.g., invalid image, model failure) are logged and reported via the context.
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Performing object detection on: {image_source}")

        if not image_source:
            ctx.error("No image source provided")
            return None

        # Fetch or process the image
        if is_url(image_source):
            async with httpx.AsyncClient() as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            return None

        

        # Check if YOLO model is preloaded
        if YOLO_MODEL is None:
            ctx.error("YOLOv8 model not available")
            logger.error("YOLOv8 model not preloaded")
            return None
        
        # Perform object detection
        try:
            with PILImage.open(BytesIO(image.data)) as img:
                logger.debug(" Started performing YOLOv8 object detection")
                results = YOLO_MODEL.predict(img, verbose=False)
                logger.debug("Completed YOLOv8 object detection")
        except Exception as e:
            ctx.error(f"Object detection failed: {str(e)}")
            logger.error(f"Detection error: {str(e)}")
            return None
        
        saved_path = None
        if save:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"
            saved_path = os.path.join(OUTPUT_DIR, filename)
            try:
                annotated_image = results[0].plot()
                annotated_pil=PILImage.fromarray(annotated_image)
                img_byte_arr=BytesIO()
                annotated_pil.save(img_byte_arr,format='JPEG')
                with open(saved_path, "wb") as f:
                    f.write(img_byte_arr.getvalue())
                logger.debug(f"Saved image to {saved_path}")
            except Exception as e:
                ctx.error(f"Failed to save image to {saved_path}: {str(e)}")
                logger.error(f"Save error: {str(e)}")
                saved_path = None

        # Process detection results
        detections = []
        labels_detected = set()
        for result in results[0].boxes:
            label = YOLO_MODEL.names[int(result.cls)]
            confidence = float(result.conf)
            bbox = result.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": {
                    "x_min": int(bbox[0]),
                    "y_min": int(bbox[1]),
                    "x_max": int(bbox[2]),
                    "y_max": int(bbox[3])
                }
            })
            labels_detected.add(label)

        # Prepare result dictionary
        result_dict = {
            "detections": detections,
            "explanation": {
                "total_detections": len(detections),
                "labels_detected": list(labels_detected),
                "image_source": image_source,
                "saved_path": saved_path if saved_path else "Not saved",
                "model_used": "yolov8n",
                "bbox_format": "Bounding box coordinates (x_min, y_min, x_max, y_max) represent the top-left and bottom-right corners of the detected object in the image.",
                "confidence_description": "Confidence scores (0.0 to 1.0) indicate the model’s certainty that the detection is correct; higher is more confident."
            }
        }

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"Object detection completed for {image_source} in {elapsed:.2f} seconds. Detected {len(detections)} objects.")
        return result_dict

    except Exception as e:
        ctx.error(f"Failed to perform object detection: {str(e)}")
        logger.error(f"Error in detect_objects_yolo: {str(e)}")
        return None
    
@mcp.tool()
async def crop_the_object(image_source: str, x_min: int, y_min: int, x_max: int, y_max: int, ctx: Context, save: bool = True) -> dict | None:
    """
    Crop a specific region of an image using provided bounding box coordinates and save it to the output directory.

    This tool is designed to be used after the `detect_objects_yolo` tool, taking the bounding box coordinates
    (x_min, y_min, x_max, y_max) from one of its detected objects to crop the corresponding region from the image.
    The cropped image is returned as an Image object and optionally saved to the 'output' directory with a
    timestamped filename in JPEG format. It reuses existing image processing logic for loading the image.

    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to the image to crop.
        x_min (int): The x-coordinate of the top-left corner of the bounding box.
        y_min (int): The y-coordinate of the top-left corner of the bounding box.
        x_max (int): The x-coordinate of the bottom-right corner of the bounding box.
        y_max (int): The y-coordinate of the bottom-right corner of the bounding box.
        ctx (Context): The MCP context object for error reporting and interaction with the server.
        save (bool): Whether to save the cropped image to the output directory (default: True).

    Returns:
        dict | None: A dictionary containing:
            - 'cropped_image': An Image object containing the cropped image data.
            - 'saved_path': The path where the cropped image was saved (if save=True, else 'Not saved').
            - 'explanation': A string describing the cropping operation and its result.
        Returns None if processing or cropping fails.

    Example:
        >>> # After running detect_objects_yolo, use its bounding box coordinates
        >>> result = await crop_the_object(
        ...     image_source="https://ultralytics.com/images/bus.jpg",
        ...     x_min=100, y_min=150, x_max=200, y_max=300,
        ...     ctx=Context(), save=True
        ... )
        >>> print(result)
        {
            'cropped_image': <Image object>,
            'saved_path': './output/cropped_image_2025-07-07_154322.jpg',
            'explanation': 'Cropped image from https://ultralytics.com/images/bus.jpg using bounding box (100, 150, 200, 300) and saved to output directory.'
        }

    """
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Cropping object from {image_source} with bounding box: ({x_min}, {y_min}, {x_max}, {y_max})")

        if not image_source:
            ctx.error("No image source provided")
            return None

        # Validate bounding box coordinates
        if x_min >= x_max or y_min >= y_max:
            ctx.error(f"Invalid bounding box coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            return None

        # Fetch or process the image
        if is_url(image_source):
            async with httpx.AsyncClient(timeout=30.0) as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            return None

        # Crop the image
        try:
            with PILImage.open(BytesIO(image.data)) as img:
                img_width, img_height = img.size
                # Clamp coordinates to image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_max)
                y_max = min(img_height, y_max)
                if x_min >= x_max or y_min >= y_max:
                    ctx.error(f"Adjusted bounding box is invalid: ({x_min}, {y_min}, {x_max}, {y_max})")
                    return None
                cropped_image = img.crop((x_min, y_min, x_max, y_max))
                img_byte_arr = BytesIO()
                cropped_image.save(img_byte_arr, format='JPEG', quality=85)
                cropped_data = img_byte_arr.getvalue()
                logger.debug(f"Cropped image to ({x_min}, {y_min}, {x_max}, {y_max})")
        except Exception as e:
            ctx.error(f"Failed to crop image: {str(e)}")
            logger.error(f"Crop error: {str(e)}")
            return None

        # Save the cropped image if requested
        saved_path = None
        if save:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"cropped_image_{timestamp}.jpg"
            saved_path = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(saved_path, "wb") as f:
                    f.write(cropped_data)
                logger.debug(f"Saved cropped image to {saved_path}")
            except Exception as e:
                ctx.error(f"Failed to save cropped image to {saved_path}: {str(e)}")
                logger.error(f"Save error: {str(e)}")
                saved_path = None

        # Prepare result dictionary
        result_dict = {
            "cropped_image": Image(data=cropped_data, format='jpeg'),
            "saved_path": saved_path if saved_path else "Not saved",
            "explanation": f"Cropped image from {image_source} using bounding box ({x_min}, {y_min}, {x_max}, {y_max}) and saved to output directory."
        }

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"Cropping completed for {image_source} in {elapsed:.2f} seconds")
        return result_dict

    except Exception as e:
        ctx.error(f"Failed to crop object: {str(e)}")
        logger.error(f"Error in crop_the_object: {str(e)}")
        return None


@mcp.tool()
async def segment_objects_yolo(image_source: str, ctx: Context,alpha: float = 0.5, save: bool = True) -> dict | None:
    """
    Perform object segmentation on an image using YOLOv11 and return the segmented image with detection details.

    This tool processes an image from a URL or local file path, performs object segmentation using the YOLOv11
    segmentation model (yolo11n-seg.pt), and overlays colored masks on detected objects with specified transparency.
    The segmented image is returned as an Image object and optionally saved to the 'output' directory with a
    timestamped filename in JPEG format. The tool reuses existing image processing logic for loading and integrates
    with other tools like `detect_objects_yolo` and `crop_the_object`.

    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to the image to segment.
        alpha (float, optional): Transparency of the segmentation masks (0.0 to 1.0, where 0.0 is fully transparent
            and 1.0 is fully opaque). Default: 0.5.
        save (bool, optional): Whether to save the segmented image to the output directory. Default: True.

    Returns:
        dict | None: A dictionary containing:
            - 'segmented_image': An Image object containing the segmented image with colored masks overlaid.
            - 'detections': A list of dictionaries, each with:
                - 'label': The class name of the detected object (e.g., 'person', 'car').
                - 'confidence': The confidence score (0.0 to 1.0) of the detection.
                - 'bbox': A dictionary with 'x_min', 'y_min', 'x_max', 'y_max' coordinates of the bounding box.
                - 'mask_area': The approximate area of the segmentation mask (in pixels).
            - 'saved_path': The path where the segmented image was saved (if save=True, else 'Not saved').
            - 'explanation': A string describing the segmentation operation, including the number of detected objects,
                transparency used, and save status.
        Returns None if processing, segmentation, or saving fails.

    Example:
        >>> result = await segment_objects_yolo(
        ...     image_source="https://ultralytics.com/images/bus.jpg",
        ...     alpha=0.5,
        ...     save=True,
        ...     ctx=Context()
        ... )
        >>> print(result)
        {
            'segmented_image': <Image object>,
            'detections': [
                {
                    'label': 'person',
                    'confidence': 0.85,
                    'bbox': {'x_min': 100, 'y_min': 150, 'x_max': 200, 'y_max': 300},
                    'mask_area': 5000
                },
                {
                    'label': 'bus',
                    'confidence': 0.92,
                    'bbox': {'x_min': 250, 'y_min': 200, 'x_max': 400, 'y_max': 350},
                    'mask_area': 15000
                }
            ],
            'saved_path': './output/segmented_image_2025-07-08_113423.jpg',
            'explanation': 'Segmented 2 objects from https://ultralytics.com/images/bus.jpg with alpha=0.5 and saved to output directory.'
        }
    """
    # Notes:
    #     - Requires `ultralytics` (`pip install ultralytics`), `numpy` (`pip install numpy`), and `opencv-python` (`pip install opencv-python`) for segmentation and image processing.
    #     - Uses the preloaded 'yolo11n-seg.pt' model for efficiency.
    #     - The segmented image includes colored masks overlaid on detected objects with the specified transparency.
    #     - Bounding box coordinates and mask areas are provided for compatibility with `crop_the_object` tool.
    #     - Errors (e.g., invalid image, model failure, timeouts) are logged and reported via the context.
    #     - The saved image is in JPEG format with quality=85, consistent with other tools.
    #     - The alpha parameter must be in the range [0.0, 1.0]; values outside this range are clamped.
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Starting object segmentation on: {image_source}, alpha: {alpha}")

        if not image_source:
            ctx.error("No image source provided")
            return None

        # Validate alpha
        alpha = max(0.0, min(1.0, float(alpha)))
        logger.debug(f"Using alpha: {alpha} for mask transparency")

        if is_url(image_source):
            async with httpx.AsyncClient(timeout=30.0) as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            return None
        
        if YOLO_MODEL is None:
            ctx.error("YOLOv11n-seg model not available")
            logger.error("YOLOv11n-seg model not preloaded")
            return None
        
        try:
            with PILImage.open(BytesIO(image.data)) as img:
                results = YOLO_MODEL_SEG.predict(img, verbose=False)
        except Exception as e:
            ctx.error(f"Object segmentation failed: {str(e)}")
            logger.error(f"Segmentation error: {str(e)}")
            return None
        

        # Process segmentation results
        detections = []
        labels_detected = set()
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
        img_np = np.array(PILImage.open(BytesIO(image.data)))  # Convert to NumPy for OpenCV processing
        img_with_masks = img_np.copy()

        # Generate random colors for each mask
        colors = [np.random.randint(0, 255, 3) for _ in range(len(masks))]

        # Overlay masks
        for i, mask in enumerate(masks):
            try:
                # Resize mask to match image dimensions
                mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask
                colored_mask = np.zeros_like(img_np)
                colored_mask[mask > 0] = colors[i]
                
                # Overlay mask with transparency
                img_with_masks = cv2.addWeighted(img_with_masks, 1.0, colored_mask, alpha, 0.0)

                # Get detection details
                result = results[0].boxes[i]
                label = YOLO_MODEL_SEG.names[int(result.cls)]
                confidence = float(result.conf)
                bbox = result.xyxy[0].tolist()
                mask_area = int(np.sum(mask))  # Approximate area in pixels
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": {
                        "x_min": int(bbox[0]),
                        "y_min": int(bbox[1]),
                        "x_max": int(bbox[2]),
                        "y_max": int(bbox[3])
                    },
                    "mask_area": mask_area
                })
                labels_detected.add(label)
            except Exception as e:
                ctx.error(f"Failed to process mask {i}: {str(e)}")
                logger.error(f"Mask processing error: {str(e)}")
                continue

        # Convert the segmented image to Image object
        try:
            segmented_pil = PILImage.fromarray(cv2.cvtColor(img_with_masks, cv2.COLOR_BGR2RGB))
            img_byte_arr = BytesIO()
            segmented_pil.save(img_byte_arr, format='JPEG', quality=85)
            segmented_data = img_byte_arr.getvalue()
            logger.debug(f"Generated segmented image with {len(detections)} masks")
        except Exception as e:
            ctx.error(f"Failed to generate segmented image: {str(e)}")
            logger.error(f"Image conversion error: {str(e)}")
            return None

        # Save the segmented image if requested
        saved_path = None
        if save:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"segmented_image_{timestamp}.jpg"
            saved_path = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(saved_path, "wb") as f:
                    f.write(segmented_data)
            except Exception as e:
                ctx.error(f"Failed to save segmented image to {saved_path}: {str(e)}")
                logger.error(f"Save error: {str(e)}")
                saved_path = None

        # Prepare result dictionary
        result_dict = {
            "segmented_image": Image(data=segmented_data, format='jpeg'),
            "detections": detections,
            "saved_path": saved_path if saved_path else "Not saved",
            "explanation": f"Segmented {len(detections)} objects from {image_source} with alpha={alpha} and saved to output directory."
        }

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"Segmentation completed for {image_source} in {elapsed:.2f} seconds. Detected {len(detections)} objects.")
        return result_dict

    except Exception as e:
        ctx.error(f"Failed to perform object segmentation: {str(e)}")
        logger.error(f"Error in segment_objects_yolo: {str(e)}")
        return None
    
@mcp.tool()
async def canny_edge_detection(image_source: str,  ctx: Context,threshold1: int = 100, threshold2: int = 200, save: bool = True) -> dict | None:
    """
    Perform Canny edge detection on an image and return the edge-detected image with details.

    This tool processes an image from a URL or local file path, applies the Canny edge detection algorithm using OpenCV,
    and returns the edge-detected image as an Image object. The result is optionally saved to the 'output' directory with a
    timestamped filename in JPEG format. The tool reuses existing image processing logic for loading and integrates with
    other tools like `detect_objects_yolo`, `crop_the_object`, and `segment_objects_yolo`.

    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to the image to process.
        threshold1 (int, optional): First threshold for the Canny edge detector (0 to 255). Controls the lower bound for
            edge detection sensitivity. Lower values detect more edges. Default: 100.
        threshold2 (int, optional): Second threshold for the Canny edge detector (0 to 255). Controls the upper bound for
            edge detection sensitivity. Higher values detect stronger edges. Default: 200.
        save (bool, optional): Whether to save the edge-detected image to the output directory. Default: True.
        ctx (Context): The MCP context object for error reporting and interaction with the server.

    Returns:
        dict | None: A dictionary containing:
            - 'edge_image': An Image object containing the edge-detected image (grayscale edges).
            - 'saved_path': The path where the edge-detected image was saved (if save=True, else 'Not saved').
            - 'explanation': A string describing the edge detection operation, including the thresholds used and save status.
        Returns None if processing or edge detection fails.

    Example:
        >>> result = await canny_edge_detection(
        ...     image_source="https://ultralytics.com/images/bus.jpg",
        ...     threshold1=100,
        ...     threshold2=200,
        ...     save=True,
        ...     ctx=Context()
        ... )
        >>> print(result)
        {
            'edge_image': <Image object>,
            'saved_path': './output/canny_image_2025-07-08_113423.jpg',
            'explanation': 'Applied Canny edge detection to https://ultralytics.com/images/bus.jpg with thresholds (100, 200) and saved to output directory.'
        }
    """
    # Notes:
    #     - Requires `opencv-python` (`pip install opencv-python`), `numpy` (`pip install numpy`), and `PIL` for image processing.
    #     - The Canny algorithm converts the image to grayscale before edge detection, producing a single-channel output.
    #     - Threshold values are clamped to the range [0, 255] to ensure compatibility with OpenCV.
    #     - The saved image is in JPEG format with quality=85, consistent with other tools.
    #     - Errors (e.g., invalid image, invalid thresholds, save failures) are logged and reported via the context.
    #     - Can be used in conjunction with `segment_objects_yolo` or `detect_objects_yolo` for preprocessing or postprocessing.
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Starting Canny edge detection on: {image_source}, threshold1: {threshold1}, threshold2: {threshold2}")

        if not image_source:
            ctx.error("No image source provided")
            return None

        # Validate thresholds
        threshold1 = max(0, min(255, int(threshold1)))
        threshold2 = max(0, min(255, int(threshold2)))
        logger.debug(f"Using thresholds: threshold1={threshold1}, threshold2={threshold2}")

        # Fetch or process the image
        if is_url(image_source):
            async with httpx.AsyncClient(timeout=30.0) as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            return None

        # Perform Canny edge detection
        try:
            with PILImage.open(BytesIO(image.data)) as img:
                # Convert to grayscale for Canny
                img_gray = img.convert('L')
                img_np = np.array(img_gray)  # Convert to NumPy array for OpenCV
                edges = cv2.Canny(img_np, threshold1, threshold2)
                # Convert edges to RGB for saving as JPEG (consistent with other tools)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                edge_pil = PILImage.fromarray(edges_rgb)
                img_byte_arr = BytesIO()
                edge_pil.save(img_byte_arr, format='JPEG', quality=85)
                edge_data = img_byte_arr.getvalue()
                logger.debug(f"Canny edge detection done ")
        except Exception as e:
            ctx.error(f"Failed to perform Canny edge detection: {str(e)}")
            logger.error(f"Edge detection error: {str(e)}")
            return None

        # Save the edge-detected image if requested
        saved_path = None
        if save:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"canny_image_{timestamp}.jpg"
            saved_path = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(saved_path, "wb") as f:
                    f.write(edge_data)
                logger.debug(f"Saved edge-detected image to {saved_path}")
            except Exception as e:
                ctx.error(f"Failed to save edge-detected image to {saved_path}: {str(e)}")
                logger.error(f"Save error: {str(e)}")
                saved_path = None

        # Prepare result dictionary
        result_dict = {
            "edge_image": Image(data=edge_data, format='jpeg'),
            "saved_path": saved_path if saved_path else "Not saved",
            "explanation": f"Applied Canny edge detection to {image_source} with thresholds ({threshold1}, {threshold2}) and saved to output directory."
        }

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.debug(f"Canny edge detection completed for {image_source} in {elapsed:.2f} seconds")
        return result_dict

    except Exception as e:
        ctx.error(f"Failed to perform Canny edge detection: {str(e)}")
        logger.error(f"Error in canny_edge_detection: {str(e)}")
        return None

@mcp.tool()
async def create_thumbnail(image_source: str, size: int, ctx: Context, save: bool = True) -> dict | None:
    """
    Create a thumbnail of an image with a specified square size, maintaining aspect ratio.

    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to the image.
        size (int): The desired width and height of the thumbnail (in pixels, e.g., 128 for 128x128).
        ctx (Context): The MCP context object for error reporting.
        save (bool): Whether to save the thumbnail to the output directory (default: True).

    Returns:
        dict | None: A dictionary containing the thumbnail image, saved path, dimensions, and explanation,
                     or None if processing fails.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        logger.debug(f"Creating thumbnail for {image_source} with size {size}x{size}")

        # Validate input parameters
        if not image_source:
            ctx.error("No image source provided")
            logger.error("No image source provided")
            return None
        size = max(16, min(512, int(size)))  # Clamp size to reasonable range
        logger.debug(f"Using thumbnail size: {size}x{size}")

        # Fetch or process the image
        fetch_start = time.time()
        if is_url(image_source):
            async with httpx.AsyncClient(timeout=60.0) as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                logger.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)
        logger.debug(f"Image fetching/processing took {time.time() - fetch_start:.2f} seconds")

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            logger.error(f"Failed to process image from {image_source}")
            return None

        # Create thumbnail
        thumbnail_start = time.time()
        try:
            with PILImage.open(BytesIO(image.data)) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                    logger.debug(f"Converted image mode to RGB for thumbnail")
                
                # Calculate thumbnail dimensions while maintaining aspect ratio
                img.thumbnail((size, size), PILImage.LANCZOS)
                width, height = img.size
                logger.debug(f"Thumbnail dimensions: {width}x{height}")

                # Save thumbnail to BytesIO
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                thumbnail_data = img_byte_arr.getvalue()
                logger.debug(f"Created thumbnail, size: {len(thumbnail_data)} bytes")
        except Exception as e:
            ctx.error(f"Failed to create thumbnail: {str(e)}")
            logger.error(f"Thumbnail creation error: {str(e)}")
            return None
        logger.debug(f"Thumbnail creation took {time.time() - thumbnail_start:.2f} seconds")

        # Save thumbnail if requested
        saved_path = None
        if save:
            save_start = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"thumbnail_{timestamp}.jpg"
            saved_path = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(saved_path, "wb") as f:
                    f.write(thumbnail_data)
                logger.debug(f"Saved thumbnail to {saved_path} in {time.time() - save_start:.2f} seconds")
            except Exception as e:
                ctx.error(f"Failed to save thumbnail to {saved_path}: {str(e)}")
                logger.error(f"Save error: {str(e)}")
                saved_path = None

        # Prepare result dictionary
        result_dict = {
            "thumbnail_image": Image(data=thumbnail_data, format='jpeg'),
            "width": width,
            "height": height,
            "saved_path": saved_path if saved_path else "Not saved",
            "explanation": f"Created a {width}x{height} thumbnail from {image_source} with target size {size}x{size} and saved to output directory."
        }

        elapsed = time.time() - start_time
        logger.debug(f"Thumbnail creation completed for {image_source} in {elapsed:.2f} seconds")
        return result_dict

    except Exception as e:
        ctx.error(f"Failed to create thumbnail: {str(e)}")
        logger.error(f"Error in create_thumbnail: {str(e)}")
        return None

@mcp.tool()
async def apply_image_filters(image_source: str, filter_type: str, ctx: Context, intensity: float = 1.0, save: bool = True) -> dict | None:
    """
    Apply various filters to an image (blur, sharpen, enhance, vintage) with adjustable intensity.

    Args:
        image_source (str): A URL (http:// or https://) or local file path pointing to the image.
        filter_type (str): The filter to apply ('blur', 'sharpen', 'enhance', 'vintage').
        ctx (Context): The MCP context object for error reporting.
        intensity (float): Intensity of the filter effect (0.0 to 2.0, default: 1.0).
        save (bool): Whether to save the filtered image to the output directory (default: True).

    Returns:
        dict | None: A dictionary containing the filtered image, saved path, and explanation,
                     or None if processing fails.
    """
    try:
        start_time = time.time()
        logger.debug(f"Applying filter '{filter_type}' to {image_source} with intensity {intensity}")

        # Validate input parameters
        if not image_source:
            ctx.error("No image source provided")
            logger.error("No image source provided")
            return None
        filter_type = filter_type.lower()
        supported_filters = {'blur', 'sharpen', 'enhance', 'vintage'}
        if filter_type not in supported_filters:
            ctx.error(f"Unsupported filter type: {filter_type}. Supported filters: {', '.join(supported_filters)}")
            logger.error(f"Unsupported filter type: {filter_type}")
            return None
        intensity = max(0.0, min(2.0, float(intensity)))  # Clamp intensity to 0.0-2.0
        logger.debug(f"Using filter: {filter_type}, intensity: {intensity}")

        # Fetch or process the image
        fetch_start = time.time()
        if is_url(image_source):
            async with httpx.AsyncClient(timeout=60.0) as client:
                image = await fetch_single_image(image_source, client, ctx)
        else:
            if not os.path.exists(image_source):
                ctx.error(f"File not found: {image_source}")
                logger.error(f"File not found: {image_source}")
                return None
            image = await process_image_data(image_source, ctx=ctx)
        logger.debug(f"Image fetching/processing took {time.time() - fetch_start:.2f} seconds")

        if image is None:
            ctx.error(f"Failed to process image from {image_source}")
            logger.error(f"Failed to process image from {image_source}")
            return None

        # Apply the selected filter
        filter_start = time.time()
        try:
            img_np = np.array(PILImage.open(BytesIO(image.data)))
            if img_np.shape[2] == 4:  # Convert RGBA to RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                logger.debug(f"Converted image to RGB for filtering")

            if filter_type == 'blur':
                # Apply Gaussian blur with kernel size based on intensity
                kernel_size = max(3, int(3 * intensity) | 1)  # Ensure odd kernel size
                filtered_img = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigmaX=intensity)
                logger.debug(f"Applied Gaussian blur with kernel size {kernel_size} and sigma {intensity}")
            elif filter_type == 'sharpen':
                # Apply sharpening filter
                kernel = np.array([[-intensity, -intensity, -intensity],
                                  [-intensity,  1 + 8*intensity, -intensity],
                                  [-intensity, -intensity, -intensity]])
                filtered_img = cv2.filter2D(img_np, -1, kernel)
                logger.debug(f"Applied sharpening filter with intensity {intensity}")
            elif filter_type == 'enhance':
                # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
                img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=intensity * 2.0, tileGridSize=(8, 8))
                img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
                filtered_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
                logger.debug(f"Applied contrast enhancement with intensity {intensity}")
            elif filter_type == 'vintage':
                # Apply sepia tone for vintage effect
                kernel = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]]) * intensity
                filtered_img = cv2.transform(img_np, kernel)
                filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
                logger.debug(f"Applied vintage sepia effect with intensity {intensity}")

            # Convert filtered image to Image object
            filtered_pil = PILImage.fromarray(filtered_img)
            img_byte_arr = BytesIO()
            filtered_pil.save(img_byte_arr, format='JPEG', quality=85)
            filtered_data = img_byte_arr.getvalue()
            logger.debug(f"Filter application took {time.time() - filter_start:.2f} seconds")
        except Exception as e:
            ctx.error(f"Failed to apply filter '{filter_type}': {str(e)}")
            logger.error(f"Filter application error: {str(e)}")
            return None

        # Save filtered image if requested
        saved_path = None
        if save:
            save_start = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"filtered_{filter_type}_{timestamp}.jpg"
            saved_path = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(saved_path, "wb") as f:
                    f.write(filtered_data)
                logger.debug(f"Saved filtered image to {saved_path} in {time.time() - save_start:.2f} seconds")
            except Exception as e:
                ctx.error(f"Failed to save filtered image to {saved_path}: {str(e)}")
                logger.error(f"Save error: {str(e)}")
                saved_path = None

        # Prepare result dictionary
        result_dict = {
            "filtered_image": Image(data=filtered_data, format='jpeg'),
            "saved_path": saved_path if saved_path else "Not saved",
            "explanation": f"Applied {filter_type} filter to {image_source} with intensity {intensity} and saved to output directory."
        }

        elapsed = time.time() - start_time
        logger.debug(f"Filter '{filter_type}' application completed for {image_source} in {elapsed:.2f} seconds")
        return result_dict

    except Exception as e:
        ctx.error(f"Failed to apply filter: {str(e)}")
        logger.error(f"Error in apply_image_filters: {str(e)}")
        return None




if __name__ == "__main__":
    mcp.run(transport='stdio')