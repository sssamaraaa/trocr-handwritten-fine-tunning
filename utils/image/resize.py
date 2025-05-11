def resize_with_aspect_and_padding(image, target_size=384, fill_color=(255, 255, 255)):
    from PIL import Image
    original_width, original_height = image.size
    ratio = target_size / max(original_width, original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))

    return new_image
