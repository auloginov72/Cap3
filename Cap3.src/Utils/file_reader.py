import re

class FileData:
    def __init__(self, columns, row_count):
        self.columns = columns
        self.row_count = row_count
    
    def get(self, column_name):
        """Get column by name"""
        if column_name in self.columns:
            return self.columns[column_name]
        else:
            print(f"Warning: Column '{column_name}' not found")
            return None
    
    def get_column_names(self):
        """Get all column names"""
        return list(self.columns.keys())
    
    def get_row_count(self):
        """Get number of rows"""
        return self.row_count
    
    def get_row(self, row_index):
        """Get specific row as dictionary"""
        if row_index < 0 or row_index >= self.row_count:
            print(f"Warning: Row index {row_index} out of bounds")
            return None
        
        row = {}
        for col_name, col_data in self.columns.items():
            row[col_name] = col_data[row_index]
        return row
    
    def get_all_data(self):
        """Get all data as list of dictionaries"""
        return [self.get_row(i) for i in range(self.row_count)]

def read_file_with_named_columns(file_path):
    """
    Read file with named columns and return FileData object
    
    Args:
        file_path (str): Absolute path to the file
        
    Returns:
        FileData: Object with methods to access data by column names
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
        
        if len(lines) < 2:
            print('Warning: File must contain at least header and one data row')
            return FileData({}, 0)
        
        # Parse header - split by tabs or multiple spaces
        header_line = lines[0]
        headers = re.split(r'[\t\s]+', header_line)
        headers = [h for h in headers if h]  # Remove empty strings
        
        if len(headers) == 0:
            print('Warning: No valid column names found in header')
            return FileData({}, 0)
        
        # Clean headers - remove % symbol for referencing
        clean_headers = [header.lstrip('%') for header in headers]
        
        # Check for unique names
        if len(set(clean_headers)) != len(clean_headers):
            print('Warning: Column names are not unique')
            return FileData({}, 0)
        
        # Parse data rows
        data_rows = []
        for i, line in enumerate(lines[1:], start=2):
            values = re.split(r'[\t\s]+', line)
            values = [v for v in values if v]  # Remove empty strings
            
            # Check if number of columns matches header length
            if len(values) != len(headers):
                print(f'Warning: Row {i} has {len(values)} columns but header has {len(headers)} columns')
                return FileData({}, 0)
            
            data_rows.append(values)
        
        # Create column data structure
        columns = {}
        for header_idx, header in enumerate(clean_headers):
            column_data = []
            for row in data_rows:
                value = row[header_idx]
                # Try to convert to number if possible
                try:
                    # Try int first, then float
                    if '.' in value or 'e' in value.lower():
                        column_data.append(float(value))
                    else:
                        column_data.append(int(value))
                except ValueError:
                    # Keep as string if conversion fails
                    column_data.append(value)
            
            columns[header] = column_data
        
        return FileData(columns=columns, row_count= len(data_rows))
        
    except Exception as error:
        print(f"Error reading file: {error}")
        return FileData({}, 0)



def read_Alarms(file_path):
    """
    Read Alarms.dat file and extract specific NewSMPL entries
    
    Args:
        file_path (str): Path to the Alarms.dat file
        
    Returns:
        FileData: Object with Time, Type, N, K columns from NewSMPL entries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
        
        # Initialize lists to store extracted data
        times = []
        types = []
        n_values = []
        k_values = []
        
        # Process each line
        for line in lines:
            # Split by tab or multiple spaces for first 4 columns
            parts = re.split(r'[\t\s]+', line, maxsplit=3)
            
            if len(parts) >= 4:
                time_str = parts[0]
                event_type = parts[1]
                
                # Check if this is a Start event
                if event_type == "Start":
                    # Get the message part (everything after the first 3 columns)
                    message = parts[3]
                    
                    # Check if it contains NewSMPL
                    if "NewSMPL;" in message:
                        # Extract the NewSMPL part
                        newsmpl_index = message.find("NewSMPL;")
                        newsmpl_part = message[newsmpl_index:]
                        
                        # Parse the NewSMPL parameters
                        # Format: NewSMPL;Type=X;N=Y;K=Z
                        type_match = re.search(r'Type=(\d+)', newsmpl_part)
                        n_match = re.search(r'N=(\d+)', newsmpl_part)
                        k_match = re.search(r'K=(\d+)', newsmpl_part)
                        
                        if type_match and n_match and k_match:
                            # Convert time to float
                            try:
                                time_value = float(time_str)
                            except ValueError:
                                print(f"Warning: Could not convert time '{time_str}' to float")
                                continue
                            
                            # Extract and convert values
                            times.append(time_value)
                            types.append(int(type_match.group(1)))
                            n_values.append(int(n_match.group(1)))
                            k_values.append(int(k_match.group(1)))
        
        # Create the data structure
        columns = {
            'Time': times,
            'Type': types,
            'N': n_values,
            'K': k_values
        }
        
        row_count = len(times)
        
        if row_count == 0:
            print("Warning: No 'Start' events with 'NewSMPL' found in file")
        else:
            print(f"Found {row_count} NewSMPL entries")
        
        return FileData(columns=columns, row_count=row_count)
        
    except Exception as error:
        print(f"Error reading alarms file: {error}")
        return FileData({}, 0)



# Example usage and testing
if __name__ == "__main__":
    # Example file content for testing:
    # Time	Crs	fXg_GPS1	fYg_GPS1	fYg.GPS2	%fYg.GPS1
    # 1.0	10.5	2.3	4.5	6.7	8.9
    # 2.0	11.5	3.3	5.5	7.7	9.9
    
    data = read_file_with_named_columns(r'C:\MatlabR15\Work\Testing\PhysInfNeurNets\InteractivePlot\___flt_gyro.dat')
    
    # Example usage:
    print("Available columns:", data.get_column_names())
    
    # Get specific columns
    time_data = data.get('Time')
    print("Time column:", time_data)
    
    # Note: % symbol is removed, so %fYg.GPS1 becomes fYg.GPS1
    fyg_data = data.get('fYg.GPS1')
    print("fYg.GPS1 column:", fyg_data)
    
    # Get a specific row
    first_row = data.get_row(0)
    print("First row:", first_row)
    
    # Get all data
    all_data = data.get_all_data()
    print("All data:", all_data)