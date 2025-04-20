import csv
import numpy as np
# References
# Kadlaskar, A. (2021, October 2). Market basket Analysis | Guide on Market Basket Analysis. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-market-basket-analysis/
# Read transactions from the CSV file
def load_data(filename):
    """Load transactions from a CSV file."""
    data = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Clean items (remove spaces, convert to lowercase)
                clean_row = [item.strip().lower() for item in row]
                data.append(clean_row)
        return data
    # Error handling
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Calculate support for given item(s)

def calc_support(data, items):
    if not data:
        return 0
    
    # Convert items to set for easier comparison {1,2,3}
    items_set = set(items)
    
    # Count transactions containing all items
    count = 0
    for transaction in data:
        if items_set.issubset(set(transaction)):
            count += 1
    
    # Calculate support = (Number of transactions containing all items) / (Total number of transactions)
    support = count / len(data)
    return support

# Calculate confidence for rule: if_items -> then_items

def calc_confidence(data, if_items, then_items):
    """Calculate confidence value for an association rule."""
    if not data:
        return 0
    
    # Convert to sets for easier comparison
    if_set = set(if_items)
    then_set = set(then_items)
    
    # Count transactions containing if_items
    if_count = 0
    for transaction in data:
        if if_set.issubset(set(transaction)):
            if_count += 1
    
    if if_count == 0:
        return 0
    
    # Count transactions containing both if_items and then_items
    both_set = if_set.union(then_set)
    both_count = 0
    for transaction in data:
        if both_set.issubset(set(transaction)):
            both_count += 1
    
    # Calculate confidence percentage Confidence = (Support of combined items) / (Support of if_items)
    confidence = both_count / if_count
    return confidence

# Parse input string into a list of items from , separated items
def split_items(items_str):
    return [item.strip().lower() for item in items_str.split(',')]

def main():
    # Path of file to analyze
    filename = "shoppingtransactions.csv"
    
    # Load transaction data
    transactions = load_data(filename)
    
    # Check if data was loaded correctly
    if not transactions:
        print("No transactions loaded. Exiting.")
        return
    
    print(f"Successfully loaded {len(transactions)} transactions.")
    
    # Main menu loop
    while True:
        print("\nAvailable commands:")
        print("1. sup item[,item]")
        print("2. con item[,item] --> item[,item]")
        print("3. exit")
        
        # Get user command
        user_input = input("\nEnter command: ").strip()
        
        # Process exit command
        if user_input.lower() == "3" or user_input.lower() == "exit":
            print("Exiting program.")
            break
        
        # Process support command
        elif user_input.lower().startswith("sup "):
            # Extract items from command
            items_text = user_input[4:].strip()
            items_list = split_items(items_text)
            
            # Calculate and display support
            support = calc_support(transactions, items_list)
            print(f"Support for {items_list}: {support:.4f} ({support*100:.2f}%)")
        
        # Process confidence command
        elif user_input.lower().startswith("con "):
            # Split rule into parts
            parts = user_input[4:].split("-->")
            
            # Check if format is correct
            if len(parts) != 2:
                print("Error: Confidence calculation requires format 'con item[,item] --> item[,item]'")
                continue
            
            # Get items from both sides of rule
            if_items = split_items(parts[0])
            then_items = split_items(parts[1])
            
            # Calculate and display confidence
            confidence = calc_confidence(transactions, if_items, then_items)
            print(f"Confidence for {if_items} --> {then_items}: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Handle unrecognized commands
        else:
            print("Command not recognized. Please try again.")

# Run the program
if __name__ == "__main__":
    main()