import sys
import json
import os
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_python as tspython

# Initialize Parsers
JAVA_LANGUAGE = Language(tsjava.language())
PYTHON_LANGUAGE = Language(tspython.language())

parser_java = Parser(JAVA_LANGUAGE)
parser_python = Parser(PYTHON_LANGUAGE)

def calculate_complexity(node):
    """
    Calculates Cyclomatic Complexity by counting branching nodes.
    """
    complexity = 1
    cursor = node.walk()
    
    visit_queue = [node]
    
    while visit_queue:
        current = visit_queue.pop(0)
        
        # Java/Python control flow keywords
        if current.type in [
            "if_statement", "for_statement", "while_statement", "case_clause", 
            "catch_clause", "conditional_expression", "for_in_statement", 
            "elif_clause", "except_clause"
        ]:
            complexity += 1
            
        # Add children
        visit_queue.extend(current.children)
            
    return complexity

def analyze_java(source_code):
    tree = parser_java.parse(bytes(source_code, "utf8"))
    root = tree.root_node
    
    structure = {
        "classes": [],
        "methods": [],
        "imports": []
    }
    
    # Simple walk to extract top-level structures
    # A real robust implementation would need recursive walking or queries
    
    cursor = root.walk()
    visited = set()
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        
        if node.id in visited:
            continue
        visited.add(node.id)
        
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                structure["classes"].append({
                    "name": name_node.text.decode("utf8"),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1
                })
                
        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                complexity = calculate_complexity(node)
                structure["methods"].append({
                    "name": name_node.text.decode("utf8"),
                    "complexity": complexity,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1
                })
        
        elif node.type == "import_declaration":
            structure["imports"].append(node.text.decode("utf8").strip())

        # Push children
        for child in node.children:
            stack.append(child)
            
    return structure

def analyze_python(source_code):
    tree = parser_python.parse(bytes(source_code, "utf8"))
    root = tree.root_node
    
    structure = {
        "classes": [],
        "functions": [],
        "imports": []
    }
    
    stack = [root]
    visited = set()
    
    while stack:
        node = stack.pop()
        if node.id in visited: continue
        visited.add(node.id)
        
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                structure["classes"].append({
                    "name": name_node.text.decode("utf8"),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1
                })
        
        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                complexity = calculate_complexity(node)
                structure["functions"].append({
                    "name": name_node.text.decode("utf8"),
                    "complexity": complexity,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1
                })
                
        elif node.type in ["import_statement", "import_from_statement"]:
            structure["imports"].append(node.text.decode("utf8").strip())
            
        for child in node.children:
            stack.append(child)
            
    return structure

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided"}))
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(json.dumps({"error": "File not found"}))
        sys.exit(1)
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
            
        if file_path.endswith(".java"):
            result = analyze_java(source)
            result["language"] = "java"
            print(json.dumps(result, indent=2))
            
        elif file_path.endswith(".py"):
            result = analyze_python(source)
            result["language"] = "python"
            print(json.dumps(result, indent=2))
            
        else:
            print(json.dumps({"error": "Unsupported file type"}))
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
