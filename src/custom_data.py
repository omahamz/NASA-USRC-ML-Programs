import csv

output_path = "shell_data_5_23.csv"

headers = ["ID", "R", "P", "A", "CC", "VC"]

amount = 200
lo = 40.25
hi = 90
step = (hi - lo) / (amount - 1)

def shell():
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        var = lo
        while var <= hi + 1e-9:  # small tolerance to include hi exactly
            row = [32.00, 4.00, 0, round(var, 6), 12, 5]
            writer.writerow(row)
            var += step
def solid():
    hi = 50
    lo = 42.5
    amount = 40
    step = (hi - lo) / (amount - 1)
    output_path = "solid_data_5_23.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        var = lo
        while var <= hi + 1e-9:  # small tolerance to include hi exactly
            row = [32.00, 4.00, 1, round(var, 6), 12, 5]
            writer.writerow(row)
            var += step

shell()