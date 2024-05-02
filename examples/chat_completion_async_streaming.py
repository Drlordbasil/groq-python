


import asyncio

from groq import AsyncGroq


async def main() -> None:
    client = AsyncGroq()

    stream = await client.chat.completions.create(
        #
        # Required parameters 
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation. This means you
            # will be explicit with its role and personality.
            {"role": "system", "content": "you are a helpful assistant."},
            # Set a user message for the assistant to respond to. This is for tasks at hand.
            {
                "role": "user",
                "content": "Explain the importance of low latency LLMs",
            },
        ],
        # The language model which will generate the completion. #
        # models: llama3-70b-8192, llama3-8b-8192	, gemma-7b-it, mixtral-8x7b-32768	UPDATED MODELS: https://console.groq.com/docs/models
        model="mixtral-8x7b-32768",
        #
        # Optional parameters
        #
        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become
        # deterministic and repetitive.
        # 0 is right on task but boringly uncreative, 0.7 is a good coding/reasoning for most.
        temperature=0.5,
        # The maximum number of tokens to generate. Requests can use up to
        # 2048 tokens shared between prompt and completion. This depends heavily on the model using. Check rate limits and generative limits: https://console.groq.com/docs/models
        max_tokens=1024,
        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,
        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        stream=True,
    )

    # Print the incremental deltas returned by the LLM.
    async for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

        # Usage information is available on the final chunk
        if chunk.choices[0].finish_reason:
            assert chunk.x_groq is not None
            assert chunk.x_groq.usage is not None
            print(f"\n\nUsage stats: {chunk.x_groq.usage}")


asyncio.run(main())
