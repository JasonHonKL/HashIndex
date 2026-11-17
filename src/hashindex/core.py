"""Core HashIndex classes and functions for document indexing and querying."""

from pathlib import Path
from typing import Dict
import json

from markitdown import MarkItDown

from hashindex.model import Model, Message
from hashindex.tools import GenerateHashKey, ListKeys, GetSummary, GetContent

# Configuration constants
MAX_TOKEN_PER_PAGE = 10000
CHARS_PER_TOKEN = 4
MAX_CHARS_PER_PAGE = MAX_TOKEN_PER_PAGE * CHARS_PER_TOKEN
MAX_CONTEXT_TOKENS = 60000
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN


class HashIndex:
    """
    A hash-based index for document pages.

    This class manages a mapping of hash keys to document pages,
    allowing for efficient retrieval and querying.
    """

    def __init__(self):
        """Initialize an empty hash index."""
        self.PageTable: Dict[str, 'HashObject'] = {}

    def to_json(self) -> str:
        """
        Convert the index to a JSON string.

        Returns:
            JSON string representation of the index.
        """
        data = {
            "pages": [
                {
                    "hash_key": key,
                    "content": obj.content,
                    "page_number": obj.page_number,
                    "summary": obj.summary,
                }
                for key, obj in self.PageTable.items()
            ]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'HashIndex':
        """
        Create an index from a JSON string.

        Args:
            json_str: JSON string representation of the index.

        Returns:
            HashIndex object.
        """
        data = json.loads(json_str)

        index = cls()
        for page_data in data["pages"]:
            index.PageTable[page_data["hash_key"]] = HashObject(
                content=page_data["content"],
                page_number=page_data["page_number"],
                summary=page_data["summary"],
            )
        return index

    def save(self, file_path: str) -> None:
        """
        Save the index to a file.

        Args:
            file_path: Path to save the index JSON file.
        """
        data = {
            "pages": [
                {
                    "hash_key": key,
                    "content": obj.content,
                    "page_number": obj.page_number,
                    "summary": obj.summary,
                }
                for key, obj in self.PageTable.items()
            ]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> 'HashIndex':
        """
        Load an index from a file.

        Args:
            file_path: Path to the index JSON file.

        Returns:
            Loaded HashIndex object.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        index = cls()
        for page_data in data["pages"]:
            index.PageTable[page_data["hash_key"]] = HashObject(
                content=page_data["content"],
                page_number=page_data["page_number"],
                summary=page_data["summary"],
            )
        return index


class HashObject:
    """
    Represents a single page in the hash index.

    Attributes:
        content: The full content of the page.
        page_number: The page number in the original document.
        summary: A generated summary of the page content.
    """

    def __init__(self, content: str, page_number: int, summary: str):
        """
        Initialize a HashObject.

        Args:
            content: The full content of the page.
            page_number: The page number.
            summary: A summary of the content.
        """
        self.content = content
        self.page_number = page_number
        self.summary = summary


def _convert_pdf_to_md(file_path: str) -> str:
    """
    Convert a PDF file to Markdown and return the path to the generated MD file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Path to the generated Markdown file.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the file is not a PDF.
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError(f"Input file must be a PDF: {file_path}")

    md_path = pdf_path.with_suffix('.md')
    md = MarkItDown()
    result = md.convert(str(pdf_path))

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(result.text_content)

    return str(md_path)


def _split_into_pages(content: str) -> list[str]:
    """
    Split content into pages of max MAX_CHARS_PER_PAGE characters.

    Be lenient with word boundaries - include the complete last word.

    Args:
        content: The content to split.

    Returns:
        List of content pages.
    """
    pages = []
    start = 0

    while start < len(content):
        end = start + MAX_CHARS_PER_PAGE

        if end < len(content):
            search_area = content[start:end]
            for i in range(len(search_area) - 1, max(0, len(search_area) - 500), -1):
                if search_area[i] in '.!?' and i + 1 < len(search_area) and search_area[i + 1] in ' \n':
                    end = start + i + 1
                    break
            else:
                for i in range(len(search_area) - 1, max(0, len(search_area) - 200), -1):
                    if search_area[i].isspace():
                        end = start + i + 1
                        break

        page_content = content[start:end].strip()
        if page_content:
            pages.append(page_content)

        start = end

    return pages


def _build_system_prompt(
    previous_hash_key: str | None,
    last_summaries: list[str],
) -> str:
    """
    Build the system prompt with context from previous pages.

    Args:
        previous_hash_key: The hash key from the previous page.
        last_summaries: List of recent page summaries.

    Returns:
        The system prompt string.
    """
    prompt = "You are analyzing document pages to create a searchable hash-based index.\n\n"
    prompt += "Your task:\n"
    prompt += "1. Generate a DESCRIPTIVE hash key that will be used to RETRIEVE this section later.\n"
    prompt += "   - The key must clearly describe WHAT content is in this section\n"
    prompt += "   - Users will query the index using natural language, so keys should be intuitive and searchable\n"
    prompt += "   - Format: lowercase_with_underscores, 3-8 specific words (e.g., 'neural_network_backpropagation')\n"
    prompt += "2. Generate a concise summary (2-4 sentences) of the content\n\n"

    if previous_hash_key or last_summaries:
        prompt += "Context from previous pages:\n"

    if previous_hash_key:
        prompt += f"- Previous hash key: {previous_hash_key}\n"

    if last_summaries:
        prompt += "- Recent page summaries:\n"
        for i, summary in enumerate(last_summaries, 1):
            prompt += f"  {i}. {summary}\n"

    prompt += "\nUse this context to ensure consistency and avoid duplicate hash keys.\n"
    prompt += "Think: 'If someone searches for this topic later, what key would help them find this section?'\n"

    return prompt


def index_pdf(
    file_path: str,
    verbose: bool = True,
    model: Model | None = None,
) -> HashIndex:
    """
    Index a PDF file using hash-based indexing.

    Args:
        file_path: Path to the PDF file to index.
        verbose: Whether to print progress messages (default: True).

    Returns:
        HashIndex object with indexed pages.
    """
    md_path = _convert_pdf_to_md(file_path)

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    pages = _split_into_pages(md_content)
    if verbose:
        print(f"Split into {len(pages)} pages")

    index = HashIndex()
    model = model if model is not None else Model()
    existing_keys: set[str] = set()

    previous_hash_key: str | None = None
    last_summaries: list[str] = []

    for page_num, page_content in enumerate(pages, start=1):
        if verbose:
            print(f"Processing page {page_num}/{len(pages)}...")

        system_prompt = _build_system_prompt(previous_hash_key, last_summaries)
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Page {page_num} content:\n\n{page_content}\n\n"
                f"Please generate a unique hash key and summary for this page.",
            },
        ]

        hash_key_tool = GenerateHashKey(existing_keys)

        response = model.complete_with_retry(
            messages=messages,
            tools=[hash_key_tool.to_dict()],
            max_retries=3,
        )

        hash_key = None
        summary = None

        if response["tool_calls"]:
            for tool_call in response["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"]["arguments"]

                if not arguments_str or not arguments_str.strip():
                    arguments_str = "{}"

                try:
                    args = json.loads(arguments_str)
                except json.JSONDecodeError:
                    continue

                if function_name == "generate_hash_key_and_summary":
                    hash_key = args.get("hash_key")
                    summary = args.get("summary")
                    try:
                        hash_key_tool(**args)
                    except ValueError as e:
                        if verbose:
                            print(f"  Validation error: {e}")
                    break

        if not hash_key:
            hash_key = f"page_{page_num}"
            while hash_key in existing_keys:
                hash_key = f"page_{page_num}_{len(existing_keys)}"
            existing_keys.add(hash_key)

        if not summary:
            summary = f"Page {page_num} content (auto-generated summary)"

        hash_obj = HashObject(
            content=page_content,
            page_number=page_num,
            summary=summary,
        )
        index.PageTable[hash_key] = hash_obj

        previous_hash_key = hash_key
        last_summaries.append(summary)
        if len(last_summaries) > 3:
            last_summaries.pop(0)

        if verbose:
            print(f"  Hash key: {hash_key}")
            print(f"  Summary: {summary[:80]}...")

    if verbose:
        print(f"Indexing complete! Total pages indexed: {len(index.PageTable)}")
    return index


def _count_tokens(messages: list[Message]) -> int:
    """
    Estimate the token count for a list of messages.
    Uses 4 characters = 1 token approximation.
    """
    total_chars = 0
    for msg in messages:
        total_chars += len(msg.get("content", ""))
        total_chars += 20  # Overhead for role and metadata
    return total_chars // CHARS_PER_TOKEN


def _compress_conversation(
    messages: list[Message],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> list[Message]:
    """
    Compress conversation by summarizing older messages if needed.

    Keeps recent messages as-is and creates a summary of older messages.

    Args:
        messages: List of messages in the conversation.
        max_tokens: Maximum tokens to keep in the conversation.

    Returns:
        Compressed list of messages.
    """
    current_tokens = _count_tokens(messages)

    if current_tokens <= max_tokens:
        return messages

    system_messages = [m for m in messages if m.get("role") == "system"]
    recent_messages = []

    tokens_count = 0
    for msg in reversed(messages):
        if msg.get("role") == "system":
            continue

        msg_tokens = (len(msg.get("content", "")) + 20) // CHARS_PER_TOKEN
        if tokens_count + msg_tokens > max_tokens * 0.8:
            break

        recent_messages.insert(0, msg)
        tokens_count += msg_tokens

    removed_count = len(messages) - len(system_messages) - len(recent_messages)
    if removed_count > 0:
        summary_msg: Message = {
            "role": "system",
            "content": f"[Note: {removed_count} earlier messages have been compressed to manage context length. "
            f"The conversation has covered previous questions and answers about the document. "
            f"Continue based on the recent context provided.]",
        }
        system_messages.append(summary_msg)

    return system_messages + recent_messages


def query_index(
    index_or_file: str | HashIndex,
    question: str,
    max_iterations: int = 10,
    verbose: bool = True,
    model: Model | None = None,
) -> str:
    """
    Query a hash index with a question.

    The agent can access summaries first, then full content as needed.
    Conversation is compressed if it exceeds MAX_CONTEXT_TOKENS.

    Args:
        index_or_file: Either a HashIndex object or path to an index JSON file.
        question: The question to answer.
        max_iterations: Maximum number of tool-using iterations (default: 10).
        verbose: Whether to print progress messages (default: True).

    Returns:
        The final answer to the question.
    """
    if isinstance(index_or_file, str):
        index = HashIndex.load(index_or_file)
    else:
        index = index_or_file

    model = model if model is not None else Model()
    tools = [ListKeys(index), GetSummary(index), GetContent(index)]

    messages: list[Message] = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions about a document using a hash index. "
            "You have access to tools that let you:\n"
            "1. list_keys - See all available pages with their summaries\n"
            "2. get_summary - Get the summary for a specific page\n"
            "3. get_content - Get the full content for a specific page\n\n"
            "Start by listing keys to understand what's available, then get summaries of relevant pages, "
            "and finally get full content when needed to answer the question in detail.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nPlease use the available tools to find the information and answer the question.",
        },
    ]

    for iteration in range(max_iterations):
        current_tokens = _count_tokens(messages)
        if current_tokens > MAX_CONTEXT_TOKENS:
            if verbose:
                print(f"Context too large ({current_tokens} tokens), compressing...")
            messages = _compress_conversation(messages)

        if verbose:
            print(f"\n=== Iteration {iteration + 1} ===")

        response = model.complete_with_retry(
            messages=messages,
            tools=[tool.to_dict() for tool in tools],
            max_retries=1,
        )

        assistant_msg: Message = {"role": "assistant"}
        if response["content"]:
            assistant_msg["content"] = response["content"]
        if response["tool_calls"]:
            assistant_msg["tool_calls"] = response["tool_calls"]
        messages.append(assistant_msg)

        if not response["tool_calls"] or not response["tool_calls"]:
            break

        for tool_call in response["tool_calls"]:
            tool_call_id = tool_call["id"]
            function_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]

            if not arguments_str or not arguments_str.strip():
                arguments_str = "{}"

            try:
                args = json.loads(arguments_str)
            except json.JSONDecodeError:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"error": "Invalid JSON in arguments"}, ensure_ascii=False),
                })
                continue

            for tool in tools:
                if tool.name == function_name:
                    try:
                        result = tool(**args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })
                        if verbose and function_name in ("get_content", "get_summary"):
                            print(f"Retrieved {function_name.split('_')[1]} for: {args.get('hash_key')}")
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                        })
                    break

    final_msg = messages[-1]
    if final_msg.get("role") == "assistant":
        return final_msg.get("content", "No response generated.")

    return "Unable to generate a response."

