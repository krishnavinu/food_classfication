from fastapi import FastAPI, Query
import json
import streamlit as st

app = FastAPI()

# Load orphanage data from JSON file
def load_orphanage_data():
    with open("orphanages.json", "r") as file:
        return json.load(file)

@app.get("/recommend")
def recommend_orphanages(item: str = Query(..., description="Item you want to donate")):
    orphanages = load_orphanage_data()
    matching_orphanages = [o for o in orphanages if item.lower() in (need.lower() for need in o["needs"])]
    
    if not matching_orphanages:
        return {"message": "No orphanages currently need this item."}
    
    return {"orphanages": matching_orphanages}

if __name__ == "__main__":
    import uvicorn
    import threading
    
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    threading.Thread(target=run_fastapi, daemon=True).start()
    
    # Streamlit UI
    st.title("Donation Recommendation System")
    item = st.text_input("Enter the item you want to donate:")
    if st.button("Find Orphanages"): 
        import requests
        response = requests.get(f"http://localhost:8000/recommend?item={item}")
        data = response.json()
        if "orphanages" in data:
            for orphanage in data["orphanages"]:
                st.write(f"**Name:** {orphanage['name']}")
                st.write(f"**Location:** {orphanage['location']}")
                st.write(f"**Needs:** {', '.join(orphanage['needs'])}")
                st.markdown(f"[Google Maps Location]({orphanage['google_maps']})", unsafe_allow_html=True)
                st.write("---")
        else:
            st.write("No orphanages currently need this item.")
