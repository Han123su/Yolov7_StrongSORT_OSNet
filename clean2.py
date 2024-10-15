input_file = "IMG_4896_MPTP_1d#3_1.txt" 
output_file = input_file.replace(".txt", "_clean.txt")

with open(input_file, 'r') as input_f:
    lines = input_f.readlines()


seen_values = set()

filtered_lines = []

for line in lines:
    parts = line.split() 
    if len(parts) >= 1:
        first_value = parts[0]
        if first_value not in seen_values:
            seen_values.add(first_value)
            filtered_lines.append(line)

with open(output_file, 'w') as output_f:
    output_f.writelines(filtered_lines)
