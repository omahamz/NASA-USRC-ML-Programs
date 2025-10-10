# NASA-USRC-ML-Programs

## Crash- and Turbulence-Resilient eVTOL Seat Base System Using Metamaterials (University of Texas, Arlington)

Studying special structures that exhibit exceptional uni-axial energy absorption and vibration damping, to enhance seat-level crashworthiness and integrate turbulence-damping technologies for electric vertical take-off and landing (eVTOL) vehicle seating.
_Student Team:_ Ryan Shrestha (Team Lead), Pablo Martinez, Omar Hamza, Taegeon Yoon, Aadi Sharma
_Faculty Mentors:_ Shiyao Lin
_Selected:_ 2025

[https://www.nasa.gov/directorates/armd/tacp/ui/usrc/usrc-awards/](url)

## Setup

```bash
git clone https://github.com/omahamz/NASA-USRC-ML-Programs.git
```

Re direct to the cloned directory

1.

```bash
python -m venv .venv
```

2.

Windows cmd:

```bash
.venv\Scripts\activate.bat
```

Windows powershell:

```bash
.venv\Scripts\Activate.ps1
```

MacOS/Linux:

```bash
source .venv/bin/activate
```

3.

```bash
pip install -r requirements.txt
```

Lastly, download 'data_folder' from the NASA Drive and locate it in the Parent directory

## Data Folder Structure

```
data_folder/ -> n elements
    1_param/ -> m elements
        {material1}/ -> k elements
            {material1}_x11.txt
            {material1}_x12.txt
            .
            .
            .
            {material1}_x1k.txt
            {material1}_params.json
        .
        .
        .
        {materialm}/ -> k elements
            {materialm}_x11.txt
            {materialm}_x12.txt
            .
            .
            .
            {materialm}_x1k.txt
            {materialm}_params.json
    2_param/ -> m elements
        {material1}/
            {material1}_x11_x21.txt
            {material1}_x12_x22.txt
            .
            .
            .
            {material1}_x1k_x2k.txt
            {material1}_params.json
        .
        .
        .
        {materialm}/ -> k elements
            {materialm}_x11.txt
            {materialm}_x12.txt
            .
            .
            .
            {materialm}_x1k.txt
            {materialm}_params.json
    .
    .
    .
    n_param/ -> m elements
        {material1}/ -> k elements
            {material1}_x11_x21..._xn1.txt
            {material1}_x12_x22_xn2.txt
            .
            .
            .
            {material1}_x1k_x2k..._xnk.txt
            {material1}_params.json
        .
        .
        .
        {material1m}/ -> k elements
            {material1}_x11_x21..._xn1.txt
            {material1}_x12_x22_xn2.txt
            .
            .
            .
            {material1}_x1k_x2k..._xnk.txt
            {material1}_params.json
    output/
        saves/
        temp/
```

### Inside `data_folder/`

Each subdirectory is named `n_param`, where `k` is the number of parameters used in the dataset.

### Inside `data_folder/n_param/`

Each `n_param` folder contains subfolders named each metamaterial type.

### Inside `data_folder/n_param/{material_name}`

Each metamaterial folder contains `.txt` data files. The filenames include the metamaterial acronym and the parameter names in the format `{material_name}..._xn.txt` and a `.json` file naming each parameter in order named `{material_name}.json` with the example format:

_Note that `.json` files are currently only formatted to handle one parameter_

```json
{
  "name": "{material_name}",
  "n": 4,
  "params": [
    "twist angle",
    "cross sectional angle",
    "outter wall thickness",
    "num of patterns"
  ],
  "min": 0,
  "max": 180
}
```

## Inside `output/`

Disclosed processed data such as graphs and advanced tables
