export async function POST(req) {
  try {
    const body = await req.json();
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return Response.json(
        { error: 'Server is missing GEMINI_API_KEY. Add it to .env.local and restart the dev server.' },
        { status: 500 }
      );
    }

    // Gemini uses "model" instead of "assistant" for the AI's turns.
    const contents = (body.messages || []).map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));

    const geminiRes = await fetch(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
        },
        body: JSON.stringify({
          system_instruction: { parts: [{ text: body.system || '' }] },
          contents,
        }),
      }
    );

    const data = await geminiRes.json();

    if (!geminiRes.ok) {
      return Response.json(
        { error: data?.error?.message || 'The AI provider returned an error.' },
        { status: geminiRes.status }
      );
    }

    const text = data?.candidates?.[0]?.content?.parts?.map((p) => p.text).join('') || '';

    // Normalize to the same {content:[{type:'text', text}]} shape the frontend already expects
    // from the Anthropic-style response — so the component file needs zero changes.
    return Response.json({
      content: [
        {
          type: 'text',
          text: text || "I couldn't put together an answer that time — try rephrasing.",
        },
      ],
    });
  } catch (err) {
    return Response.json(
      { error: 'Something went wrong reaching the AI advisor.' },
      { status: 500 }
    );
  }
}