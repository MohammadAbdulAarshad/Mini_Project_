import streamlit as st
import pickle as p

# Load the model and feature transformer
def load_artifacts():
    with open(r"C:\Mini_Project_\model.pkl", 'rb') as model_file:
        model = p.load(model_file)
    with open(r"C:\Mini_Project_\feature.pkl", 'rb') as feature_file:
        fea = p.load(feature_file)
    return model, fea

model, fea = load_artifacts()

# App Title and Description
st.set_page_config(page_title="Fraud Mail Detection App", layout="centered")
st.markdown(
    """
    <div style='background-color:#4A90E2;padding:15px;border-radius:10px;text-align:center;'>
        <h1 style='color:#FFFFFF;'>📧 Fraud Mail Detection App</h1>
    </div>
    """, 
    unsafe_allow_html=True
)
st.write(
    """
    <div style='background-color:#D1C4E9;padding:15px;border-radius:10px;margin-top:10px;'>
        <p style='color:#4527A0;'>💡 Enter the content of your email below to determine whether it's a **spam** or **legitimate** email. Our AI model will analyze it for you!</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Input Section
user_input = st.text_area("✉️ **Enter your Mail:**", placeholder="Type or paste the email content here...")

# Prediction Section
if st.button("🚀 **Detect Spam**"):
    if not user_input.strip():
        st.warning("⚠️ **Please enter some text before submitting!**")
    else:
        # Transform and predict
        inp = fea.transform([user_input])
        result = model.predict(inp)

        # Display results
        if result == 0:
            st.markdown(
                """
                <div style='background-color:#FFCDD2;padding:20px;border-radius:10px;margin-top:10px;'>
                    <h2 style='color:#D32F2F;'>🚫 Spam Detected!</h2>
                    <p style='color:#B71C1C;'>🔴 The email you entered appears to be spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style='background-color:#C8E6C9;padding:20px;border-radius:10px;margin-top:10px;'>
                    <h2 style='color:#388E3C;'>✅ Legitimate Email!</h2>
                    <p style='color:#1B5E20;'>🟢 The email you entered is not spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Footer
st.markdown(
    """
    <div style='background-color:#FFECB3;padding:15px;border-radius:10px;margin-top:20px;text-align:center;'>
        <p style='color:#795548;'>🌟 Thank you for using the Fraud Mail Detection App! Stay spam-free! 🌟</p>
    </div>
    """, 
    unsafe_allow_html=True
)
