"""
Normalize atomic database names and version identifiers, and generate the
metadata table used by the database list interface.
"""

import os


def database_convert(s):
    """Extract the database name and numeric version identifier from a string."""
    s = s.lower()

    # Read the leading database name, stopping at a non-letter or the version marker.
    db_name_chars = []
    for ch in s:
        if ch.isalpha() and ch != 'v':
            db_name_chars.append(ch)
        else:
            break
    db_name = ''.join(db_name_chars)

    # Collect all numeric characters in the full input string.
    version_digits = [ch for ch in s if ch.isdigit()]
    version = ''.join(version_digits)

    return db_name, version, db_name + version


def main():
    """Generate the database metadata list used by the application."""
    input_list = ['AtomDB v3.0.9', 'CHIANTI v10.0.1', 'SPEX v3.06.01']
    latest_list = [0, 0, 0]

    _, _, db_list = zip(*[database_convert(item) for item in input_list])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    target_dir = os.path.join(parent_dir, "data")
    file_path = os.path.join(target_dir, 'list_database.dat')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('db\tdb_note\tlatest\n')
        for db, db_note, latest in zip(db_list, input_list, latest_list):
            file.write(f"{db}\t{db_note}\t{latest}\n")


if __name__ == "__main__":
    main()
