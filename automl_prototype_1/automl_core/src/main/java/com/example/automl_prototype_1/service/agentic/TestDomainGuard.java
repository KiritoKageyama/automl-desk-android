package com.example.automl_prototype_1.service.agentic;

public class TestDomainGuard {
    public static void main(String[] args) {
        DomainGuardService guard = new DomainGuardService();

        String[] tests = {
                // --- CLEARLY ALLOWED ---
                "Explain QuickSort algorithm time complexity",
                "Debug this NullPointerException in Java",
                "Write a Python script for matrix multiplication using NumPy",
                "System architecture for a distributed microservices app",
                "What is the difference between TCP and UDP?",
                "Implement a thread-safe Singleton pattern in C#",
                "How does a neural network backpropagation work?",
                "Configure a Docker container for a Flask app",
                "Explain the concept of virtual memory in OS",
                "Best practices for preventing SQL injection attacks",

                // --- CLEARLY BLOCKED ---
                "How to bake a chocolate cake?",
                "Who won the Super Bowl last year?",
                "What is the capital of France?",
                "Write a poem about a lonely flower",
                "Tips for training a puppy",
                "The weather is nice today",
                "History of the Roman Empire",
                "Medical advice for a headache",

                // --- EDGE CASES / AMBIGUOUS ---
                "What is a virus?", // Biological or Computer?
                "How to kill a process?", // morbid or technical?
                "Define a tree", // Nature or Data Structure?
                "What is a class?", // School or OOP?
                "Explain the concept of inheritance", // Biology or OOP?
                "Is Python dangerous?", // Snake or Language?
                "How to structure a paragraph?", // Writing or Code comments?

                // --- CONVERSATIONAL / JARGON ---
                "Can you help me with my code?",
                "I have a bug",
                "My server is down",
                "git push force",
                "rm -rf /",
                "select * from users",
                "<html><body></body></html>"
        };

        System.out.println("--- Domain Guard Verification ---");
        for (String test : tests) {
            boolean allowed = guard.validate(test);
            DomainGuardService.ClassificationResult result = guard.classify(test);
            System.out.printf("[%s] '%s' -> %s (Score: %.2f)%n",
                    allowed ? "ALLOWED" : "BLOCKED",
                    test,
                    result.domain,
                    result.score);
        }
    }
}
