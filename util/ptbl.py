"""
Utilities for converting between atomic numbers, element symbols, and element names.
The script can also generate a periodic-table mapping file for isoelectronic sequences.
"""
import os

# Periodic-table mapping from atomic number to element symbol.
num_to_element = {
    1: "H",   2: "He",  3: "Li",  4: "Be",  5: "B",
    6: "C",   7: "N",   8: "O",   9: "F",   10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S",  17: "Cl", 18: "Ar", 19: "K",  20: "Ca",
    21: "Sc", 22: "Ti", 23: "V",  24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br",
    36: "Kr", 37: "Rb", 38: "Sr", 39: "Y",  40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh",
    46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I",  54: "Xe", 55: "Cs",
    56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb",
    66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W",  75: "Re",
    76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U",  93: "Np", 94: "Pu", 95: "Am",
    96: "Cm", 97: "Bk", 98: "Cf"
}

# Mapping from lowercase element names to atomic numbers.
element_fullname_to_num = {
    'hydrogen': 1, 'helium': 2, 'lithium': 3, 'beryllium': 4, 'boron': 5, 'carbon': 6, 'nitrogen': 7,
    'oxygen': 8, 'fluorine': 9, 'neon': 10, 'sodium': 11, 'magnesium': 12, 'aluminum': 13, 'silicon': 14,
    'phosphorus': 15, 'sulfur': 16, 'chlorine': 17, 'argon': 18, 'potassium': 19, 'calcium': 20, 'scandium': 21,
    'titanium': 22, 'vanadium': 23, 'chromium': 24, 'manganese': 25, 'iron': 26, 'cobalt': 27, 'nickel': 28,
    'copper': 29, 'zinc': 30, 'gallium': 31, 'germanium': 32, 'arsenic': 33, 'selenium': 34, 'bromine': 35,
    'krypton': 36, 'rubidium': 37, 'strontium': 38, 'yttrium': 39, 'zirconium': 40, 'niobium': 41,
    'molybdenum': 42, 'technetium': 43, 'ruthenium': 44, 'rhodium': 45, 'palladium': 46, 'silver': 47,
    'cadmium': 48, 'indium': 49, 'tin': 50, 'antimony': 51, 'tellurium': 52, 'iodine': 53, 'xenon': 54,
    'cesium': 55, 'barium': 56, 'lanthanum': 57, 'cerium': 58, 'praseodymium': 59, 'neodymium': 60,
    'promethium': 61, 'samarium': 62, 'europium': 63, 'gadolinium': 64, 'terbium': 65, 'dysprosium': 66,
    'holmium': 67, 'erbium': 68, 'thulium': 69, 'ytterbium': 70, 'lutetium': 71, 'hafnium': 72,
    'tantalum': 73, 'tungsten': 74, 'rhenium': 75, 'osmium': 76, 'iridium': 77, 'platinum': 78,
    'gold': 79, 'mercury': 80, 'thallium': 81, 'lead': 82, 'bismuth': 83, 'polonium': 84, 'astatine': 85,
    'radon': 86, 'francium': 87, 'radium': 88, 'actinium': 89, 'thorium': 90, 'protactinium': 91,
    'uranium': 92, 'neptunium': 93, 'plutonium': 94, 'americium': 95, 'curium': 96, 'berkelium': 97,
    'californium': 98, 'einsteinium': 99, 'fermium': 100, 'mendelevium': 101, 'nobelium': 102,
    'lawrencium': 103, 'rutherfordium': 104, 'dubnium': 105, 'seaborgium': 106, 'bohrium': 107,
    'hassium': 108, 'meitnerium': 109, 'darmstadtium': 110, 'roentgenium': 111, 'copernicium': 112,
    'nihonium': 113, 'flerovium': 114, 'moscovium': 115, 'livermorium': 116, 'tennessine': 117,
    'oganesson': 118
}

# Conversion utilities.
def atomic_number_to_element(atomic_number):
    return num_to_element.get(atomic_number, None)

def element_to_atomic_number(element):
    for number, sym in num_to_element.items():
        if sym.lower() == element.lower():
            return number
    return None

def element_fullname_to_atomic_number(element_name):
    """Convert an element name to lowercase and return its atomic number, e.g. Iron -> 26."""
    element_name = element_name.lower()  # Normalize the input name to lowercase.
    return element_fullname_to_num.get(element_name, None)

# Convert an atomic number to a capitalized element name.
def atomic_number_to_element_fullname(atomic_number):
    """Return the capitalized element name corresponding to an atomic number."""
    num_to_element_fullname = {v: k for k, v in element_fullname_to_num.items()}
    name = num_to_element_fullname.get(atomic_number)
    if name:
        return name.capitalize()
    else:
        return None

# Generate the periodic-table mapping file.
def main():
    # Resolve the output directory relative to this script.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    target_dir = os.path.join(parent_dir, "data")
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, "ptbl.dat")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("n_bound\tseq\n")
        for number in sorted(num_to_element.keys()):
            element = num_to_element[number] + "-like"
            file.write(f"{number}\t{element}\n")

if __name__ == "__main__":
    main()
