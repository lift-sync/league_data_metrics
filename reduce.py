rows = []

with open('rockyou.txt', encoding='latin-1') as text:
    for line in text:
        password = line.strip()
        if(len(rows) < 100):
            rows.append(password)
        else:
            break

with open("salida.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(map(str, rows)))