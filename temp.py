def save_utf8_bom(filename):
  """
  Reads a file, encodes it as UTF-8 with BOM, and saves it back.
  """
  with open(filename, 'r') as f:
    content = f.read()
  with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)
    print(f"Saved '{filename}' with UTF-8 encoding (BOM).")

# Replace 'requirements.txt' with your actual filename
save_utf8_bom('requirements.txt')
