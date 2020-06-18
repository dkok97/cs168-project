import csv

ids = []
dates = []
pathologies = []

with open('../raw_pathology.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        ids.append(row[0])
        dates.append(row[1])
        if row[4][:5].lower() == 'clear':
            pathologies.append('malignant')
        else:
            pathologies.append('benign')

print(len(ids))
print(len(dates))
print(len(pathologies))

with open('../pathology.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['id', 'pathology', 'date'])
    for i in range(1, len(ids)):
        writer.writerow([ids[i], dates[i], pathologies[i]])