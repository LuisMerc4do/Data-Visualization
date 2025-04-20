import csv
import numpy as np

def read_transactions(filename):
    """Read transactions from a CSV file."""
    transactions = []
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Clean items by removing spaces and converting to lowercase
                clean_row = [item.strip().lower() for item in row]
                transactions.append(clean_row)
        return transactions
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def calculate_support(transactions, items):
    """Calculate support for given item(s)."""
    if not transactions:
        return 0
    
    # Convert string items to set for comparison
    items_set = set(items)
    
    # Count transactions that contain all items
    count = sum(1 for transaction in transactions if items_set.issubset(set(transaction)))
    
    # Calculate support
    support = count / len(transactions)
    return support

def calculate_confidence(transactions, antecedent, consequent):
    """Calculate confidence for rule: antecedent -> consequent."""
    if not transactions:
        return 0
    
    # Convert to sets for comparison
    antecedent_set = set(antecedent)
    consequent_set = set(consequent)
    
    # Count transactions containing antecedent
    antecedent_count = sum(1 for transaction in transactions if antecedent_set.issubset(set(transaction)))
    
    if antecedent_count == 0:
        return 0
    
    # Count transactions containing both antecedent and consequent
    combined_set = antecedent_set.union(consequent_set)
    combined_count = sum(1 for transaction in transactions if combined_set.issubset(set(transaction)))
    
    # Calculate confidence
    confidence = combined_count / antecedent_count
    return confidence

def parse_items(items_str):
    """Parse a comma-separated items string into a list."""
    return [item.strip().lower() for item in items_str.split(',')]

def main():
    filename = "shoppingtransactions.csv"
    transactions = read_transactions(filename)
    
    if not transactions:
        print("No transactions loaded. Exiting.")
        return
    
    print(f"Successfully loaded {len(transactions)} transactions.")
    
    while True:
        print("\nAvailable commands:")
        print("1. sup item[,item]")
        print("2. con item[,item] --> item[,item]")
        print("3. exit")
        
        choice = input("\nEnter command: ").strip()
        
        # Exit command
        if choice.lower() == "3" or choice.lower() == "exit":
            print("Exiting program.")
            break
        
        # Support command
        elif choice.lower().startswith("sup "):
            items_str = choice[4:].strip()
            items = parse_items(items_str)
            
            support = calculate_support(transactions, items)
            print(f"Support for {items}: {support:.4f} ({support*100:.2f}%)")
        
        # Confidence command
        elif choice.lower().startswith("con "):
            rule_parts = choice[4:].split("-->")
            
            if len(rule_parts) != 2:
                print("Error: Confidence calculation requires format 'con item[,item] --> item[,item]'")
                continue
            
            antecedent = parse_items(rule_parts[0])
            consequent = parse_items(rule_parts[1])
            
            confidence = calculate_confidence(transactions, antecedent, consequent)
            print(f"Confidence for {antecedent} --> {consequent}: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Command not recognized
        else:
            print("Command not recognized. Please try again.")

if __name__ == "__main__":
    main()