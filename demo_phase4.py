#!/usr/bin/env python3
"""
Conversational RAG Interface - Interactive Demo

A terminal-based chat UI for the Phase 4 Conversational RAG system.

Usage:
    python3 demo_phase4.py

Commands:
    /help     - Show available commands
    /summary  - Show conversation summary
    /topics   - Show topics timeline
    /stats    - Show conversation statistics
    /clear    - Clear conversation history
    /quit     - Exit the demo
"""

import sys
from src.conversational_rag_interface import create_conversational_rag_interface


def print_header():
    """Print the welcome header."""
    print("\n" + "=" * 60)
    print("  Conversational RAG Interface - Phase 4 Demo")
    print("=" * 60)
    print("\nType your questions to chat with the RAG system.")
    print("Type /help for available commands.\n")


def print_help():
    """Print available commands."""
    print("\n--- Available Commands ---")
    print("  /help     - Show this help message")
    print("  /summary  - Show conversation summary")
    print("  /topics   - Show topics discussed timeline")
    print("  /stats    - Show conversation statistics")
    print("  /history  - Show recent conversation history")
    print("  /clear    - Clear conversation history")
    print("  /quit     - Exit the demo")
    print("-" * 30 + "\n")


def print_summary(interface):
    """Print conversation summary."""
    summary = interface.get_summary()
    print("\n--- Conversation Summary ---")
    print(f"Turns: {summary['turn_count']}")
    print(f"Main Topics: {', '.join(summary['main_topics']) if summary['main_topics'] else 'None yet'}")
    if summary['key_points']:
        print("Key Points:")
        for i, point in enumerate(summary['key_points'][:3], 1):
            print(f"  {i}. {point[:80]}...")
    print(f"\nSummary: {summary['summary_text']}")
    print("-" * 30 + "\n")


def print_topics(interface):
    """Print topics timeline."""
    timeline = interface.get_topics_timeline()
    print("\n--- Topics Timeline ---")
    if not timeline:
        print("No topics discussed yet.")
    else:
        for entry in timeline[:10]:
            topics = ", ".join(entry['new_topics'])
            print(f"  [{entry['role']}] New topics: {topics}")
    print("-" * 30 + "\n")


def print_stats(interface):
    """Print conversation statistics."""
    stats = interface.get_statistics()
    print("\n--- Conversation Statistics ---")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Follow-up Queries: {stats['follow_up_queries']}")
    print(f"  Clarification Queries: {stats['clarification_queries']}")
    print(f"  New Topic Queries: {stats['new_topic_queries']}")
    print(f"  Total Turns: {stats['total_turns']}")
    print(f"  User Messages: {stats['user_messages']}")
    print(f"  Assistant Messages: {stats['assistant_messages']}")
    print("-" * 30 + "\n")


def print_history(interface):
    """Print recent conversation history."""
    history = interface.get_conversation_history(recent_only=True, count=6)
    print("\n--- Recent Conversation ---")
    if not history:
        print("No conversation history yet.")
    else:
        for turn in history:
            role = turn['role'].upper()
            content = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
            print(f"  [{role}]: {content}")
    print("-" * 30 + "\n")


def format_response(result):
    """Format the chat response for display."""
    print("\n" + "-" * 40)
    print(f"Assistant: {result['response']}")
    print("-" * 40)

    # Show query analysis
    analysis = result['query_analysis']
    query_type = analysis['type']
    is_follow_up = analysis['is_follow_up']

    info_parts = [f"Query type: {query_type}"]
    if is_follow_up:
        info_parts.append("(follow-up)")
    if analysis.get('topics'):
        info_parts.append(f"Topics: {', '.join(analysis['topics'][:3])}")

    print(f"[{' | '.join(info_parts)}]")
    print()


def load_sample_knowledge(interface):
    """Load sample knowledge base content."""
    knowledge_items = [
        "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It allows language models to access external knowledge bases to provide more accurate and up-to-date responses.",

        "The main benefits of RAG systems include: 1) Access to current information beyond the model's training data, 2) Reduced hallucination through grounded responses, 3) More cost-effective than fine-tuning for many use cases, 4) Easy to update knowledge without retraining.",

        "Vector embeddings are numerical representations of text that capture semantic meaning. Similar concepts have similar embeddings, enabling semantic search rather than just keyword matching.",

        "Hybrid retrieval combines multiple search strategies: semantic similarity using embeddings, keyword matching using TF-IDF, and metadata filtering. This approach often outperforms single-strategy retrieval.",

        "Fine-tuning involves training a model on specific data to adapt it for particular tasks. While powerful, it's more expensive and requires technical expertise compared to RAG approaches.",

        "Query expansion improves retrieval by adding synonyms and related terms to user queries. For example, 'ML' might be expanded to include 'machine learning' and 'artificial intelligence'.",

        "Conversation memory in RAG systems allows the model to maintain context across multiple turns, enabling natural follow-up questions and references to earlier parts of the conversation.",
    ]

    for item in knowledge_items:
        interface.add_knowledge(item)

    print(f"Loaded {len(knowledge_items)} knowledge items.\n")


def main():
    """Main function to run the interactive demo."""
    print_header()

    # Create the conversational RAG interface
    interface = create_conversational_rag_interface()

    # Load sample knowledge
    print("Loading sample knowledge base...")
    load_sample_knowledge(interface)

    print("Ready! Start chatting or type /help for commands.\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command == "/quit" or command == "/exit":
                    print("\nGoodbye! Thanks for using Conversational RAG.\n")
                    break
                elif command == "/help":
                    print_help()
                elif command == "/summary":
                    print_summary(interface)
                elif command == "/topics":
                    print_topics(interface)
                elif command == "/stats":
                    print_stats(interface)
                elif command == "/history":
                    print_history(interface)
                elif command == "/clear":
                    interface.clear_history()
                    print("\nConversation history cleared.\n")
                else:
                    print(f"\nUnknown command: {user_input}")
                    print("Type /help for available commands.\n")
                continue

            # Process the chat message
            result = interface.chat(user_input)
            format_response(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.\n")
        except EOFError:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
