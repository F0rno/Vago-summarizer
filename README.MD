# Video Summarizer

Video Summarizer is a Python-based application that processes video files to generate markdown summaries. It uses the [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) to access local llms to summarize videos into markdown format.

## Example

This is the transcription from a game theory course from Coursera:

```
One, two, three. One, two, three. One, two, three. The rhythmic counting continued as Matt shuffled the deck of cards, his fingers moving deftly through the stack. Kevin glanced around the room, a curious expression on his face. "Hey guys, what are you doing?" he asked, breaking the silence. Matt looked up and smiled, his eyes glinting with amusement. "Oh, hello, I've got a friend over here," he replied, gesturing toward Yov, who was sitting across from him at the small table. "Who are you?" Kevin inquired, turning his attention to the newcomer. "I'm Kevin." Yov grinned and leaned back in his chair. "Oh, nice, I'm Yov," he said, extending a hand. Matt chuckled and added, "I'm Matt." The three exchanged brief pleasantries, their initial awkwardness quickly giving way to a more relaxed atmosphere. "I guess we ought to play a new game, huh?" Kevin suggested, glancing at the deck of cards in Matt's hands.

Yov nodded thoughtfully. "Um... What do you suggest?" he asked, genuinely curious. Matt's eyes lit up with an idea. "Ah, how about some Blackjack?" he proposed, shuffling the cards one last time. Kevin's face brightened, and he nodded enthusiastically. "I could do that," he agreed. However, Yov seemed a bit hesitant. "I don't have an office. I'm in office, so let's play for low stakes, okay? Like $10,000?" he joked, causing the others to laugh. "Oh, it's a deal," Matt replied with a grin, clearly enjoying the light-hearted banter. They settled into their seats, ready to begin the game. "Alright. Let's see who wins."
```

And this is the markdown from the video summarizer:

```markdown
**Welcome to Game Theory: A Course Overview**
=====================================

**What is Game Theory?**
------------------------

Game theory is a branch of social sciences that studies strategic interactions among self-interested agents. While we often think about games like rock-paper-scissors or poker, game theory encompasses a broader range of interactions, including auctions, financial markets, international conflict, and more.

**The Intersection of Computer Science and Game Theory**
---------------------------------------------------------

As computer science has become increasingly important in bringing people together, it requires modeling self-interested participants and their strategic interactions. Similarly, many economic problems have computational dimensions that require leveraging ideas from algorithms, complexity, and artificial intelligence to make them work in practice.

**Our Course: A Seven-Week Introduction**
----------------------------------------

This course is an introductory class that assumes no prior experience with game theory. We'll cover the fundamental modeling tools, including equilibrium notions, incomplete information, timing repeated games, and co-illumination structures. Our intention is not to advise on real-world applications but rather to illustrate formal constructs using examples.

**Course Structure**
-------------------

*   Ungraded quizzes to test comprehension after videos
*   Review quizzes at the end of each concession
*   Lab exercises where you can play games with other students and discuss them afterwards
*   Problem sets and a final exam, which you'll only be able to take once, but you'll be able to take on your own schedule
*   A certificate of completion signed by all instructors

**Class Expectations**
----------------------

This is a large class, and we encourage you to form study groups to consult each other on problem sets, lectures, and online forums. While we'll monitor the forums, it's essential that you work together as a community to make this course successful.

### **Key Points**

*   Game theory studies strategic interactions among self-interested agents.
*   The intersection of computer science and game theory requires modeling self-interested participants and their strategic interactions.
*   This course is an introductory class that assumes no prior experience with game theory.
*   We'll cover the fundamental modeling tools, including equilibrium notions, incomplete information, timing repeated games, and co-illumination structures.
*   The course structure includes ungraded quizzes, review quizzes, lab exercises, problem sets, and a final exam.
```

## Features

- Extracts audio from videos
- Generates transcriptions from extracted audio
- Generates summaries from the audio transcriptions
- Local LLMs by the Ollama API

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/video-summarizer.git
   ```
2. Navigate to the project directory:
   ```sh
   cd video-summarizer
   ```
3. Create a Python environment (optional) and install the requirements
   ```sh
   conda create --name video-summarizer python=3.8 -y
   conda activate video-summarizer
   pip install -r requirements.txt
   ```

## Usage

To summarize a video, run the following command:

```sh
python video-summarizer.py --input /path/to/video/file --output /path/to/output/file --info
```

### Additional Options

+ --output-path: Specify the path for the summary file (default is ./).
+ --system-prompt: Custom system prompt for summarization.
+ --model: Specify the model name of the API (default is llama3.1).
+ --no-cache: Do not save intermediate files, slowing down the summary iteration.
+ --purge-all-cache: Delete all the intermediate files.
+ --info: Print logs.
+ --tool-mode: The program will not summarize the video but will let you use tool parameters.
+ --tool-count-tokens-from-file: Requires --tool-mode flag. Tells you how many tokens a file has plus your system prompt to know if it is too long for your model.
+ --tool-N-chars-to-tokens: Requires --tool-mode flag. Output the number of tokens equivalent to the number of characters inputted.
