import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
from groq import Groq
import os
import json

# Get API key from Streamlit secrets or environment variable
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please configure it in Streamlit secrets or as an environment variable.")
    st.markdown("""
    **To fix this:**
    1. **Set Environment Variable**:
       - In PyCharm: Go to `Run > Edit Configurations`, add `GROQ_API_KEY=gsk_...YceG` under Environment variables.
       - In terminal: Run `export GROQ_API_KEY='gsk_...YceG'` (Linux/Mac) or `set GROQ_API_KEY=gsk_...YceG` (Windows).
    2. **Use Streamlit Secrets**:
       - Create a `secrets.toml` file in your project directory (or `.streamlit/secrets.toml`) with:
         ```toml
         GROQ_API_KEY = "gsk_...YceG"
         ```
       - If deployed on Streamlit Cloud, add `GROQ_API_KEY` in the app's secrets settings.
    3. Replace `gsk_...YceG` with your full API key from Groq.
    """)
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}. Please verify your GROQ_API_KEY.")
    st.stop()

# Define base prices
base_prices = {
    'Paneer': 250, 'Milk': 70, 'Onions': 60,
    'Potatoes': 55, 'Meat': 320, 'Oil': 100
}

# Product data
products_data = {
    'Product': ['Paneer', 'Milk', 'Onions', 'Potatoes', 'Meat', 'Oil'],
    'Available_Stock': [100, 200, 150, 180, 120, 250],
    'Minimum_Level': [20, 30, 40, 50, 30, 70],
    'Lead_Time_days': [3, 2, 4, 5, 3, 7],
    'Max_Stock': [100, 200, 150, 180, 120, 250]
}

demand_rate = {'Paneer': 10, 'Milk': 15, 'Onions': 8, 'Potatoes': 12, 'Meat': 5, 'Oil': 20}
units = {'Paneer': 'kgs', 'Milk': 'packets', 'Onions': 'kgs', 'Potatoes': 'kgs', 'Meat': 'kgs', 'Oil': 'kgs'}

# Product descriptions
product_descriptions = {
    'Paneer': "Our Paneer is super fresh, creamy, and perfect for your curries or grilled dishes!",
    'Milk': "This Milk is pure, rich, and sourced from happy cows – ideal for your daily needs!",
    'Onions': "Our Onions are crisp, juicy, and add the perfect zing to any dish!",
    'Potatoes': "These Potatoes are fresh, versatile, and great for fries, curries, or mashed delights!",
    'Meat': "Our Meat is tender, high-quality, and freshly cut for your delicious meals!",
    'Oil': "This Oil is pure, healthy, and perfect for cooking or frying your favorite dishes!"
}

# Initialize inventory
inventory_df = pd.DataFrame(products_data)
inventory_df['Demand_Rate_per_Day'] = inventory_df['Product'].map(demand_rate)
inventory_df['ROL'] = inventory_df.apply(
    lambda row: (row['Demand_Rate_per_Day'] * row['Lead_Time_days']) + row['Minimum_Level'], axis=1)
inventory_df['Units'] = inventory_df['Product'].map(units)

# --- Session State Initialization ---
for key, value in {
    'stock': inventory_df.copy(),
    'order_items': [],
    'placed_orders': [],
    'order_count': 1,
    'supplier_quotes': {},
    'procurement_orders': [],
    'update_trigger': 0,
    'pending_confirmation': None,
    'order_placed': {},
    'simulation_day': datetime.today().date(),
    'day_started': False,
    'supplier_performance': {},
    'checked_rows': set(),
    'sent_reminders': {},
    'chat_history': [],
    'chat_input_counter': 0,
    'last_user_input': None,
    'show_chat': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Title
st.title(f"🛒 Product Ordering App - {st.session_state.simulation_day.strftime('%Y-%m-%d')}")

# --- CUSTOMER CHATBOT ---
def process_query(query, stock_df, base_prices, units, product_descriptions, client):
    """Process user query with Groq's Llama 3.1 70B."""
    if not query.strip():
        return None, None, None

    products = stock_df['Product'].tolist()
    stock_info = stock_df.set_index('Product')[['Available_Stock', 'ROL']].to_dict('index')
    context = {
        'products': products,
        'descriptions': product_descriptions,
        'base_prices': base_prices,
        'units': units,
        'stock_info': stock_info
    }

    prompt = f"""
    You are a friendly customer support chatbot for Village Gourmet, a grocery store. Your goal is to assist users with queries about products ({', '.join(products)}), provide information, and guide them to place orders if needed. Respond in a casual, engaging tone. Based on the user's query, identify the intent and product (if any), and generate a natural response.

    Context:
    - Products: {json.dumps(products)}
    - Descriptions: {json.dumps(product_descriptions)}
    - Prices: {json.dumps(base_prices)}
    - Units: {json.dumps(units)}
    - Stock Info: {json.dumps(stock_info)}

    Intents to detect:
    - description: Asking about a product (e.g., "Tell me about Paneer")
    - availability: Checking stock (e.g., "Do you have Milk?")
    - price: Asking cost (e.g., "How much is Onions?")
    - greeting: Casual greetings (e.g., "Hello", "Hi")
    - all_products: Listing products (e.g., "What items do you have?")
    - delivery: Delivery info (e.g., "When can you deliver?")
    - unclear: Ambiguous or unrelated (e.g., "What's up?")

    Instructions:
    - Identify the product (case-insensitive) and intent from the query.
    - For 'description', use the provided description.
    - For 'availability', check if Available_Stock >= ROL (available) or not.
    - For 'price', provide the price per unit.
    - For 'all_products', list all products.
    - For 'delivery', mention 2-3 day delivery.
    - For 'greeting', welcome and suggest asking about products.
    - For 'unclear', guide to ask about products or ordering.
    - Return a JSON object with: intent (string), product (string or null), response (string).

    Query: "{query}"
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=200
        )
        result = json.loads(completion.choices[0].message.content)
        return result.get('response'), result.get('intent'), result.get('product')
    except Exception as e:
        st.error(f"Groq API error: {str(e)}. Please check your API key or network connection.")
        return "Sorry, I'm having trouble processing your query. Please try again!", "unclear", None

st.sidebar.header("💬 Customer Support")
if st.sidebar.button("Chat with Us", key="chat_toggle"):
    st.session_state.show_chat = not st.session_state.show_chat
    if st.session_state.show_chat:
        # Initialize chat with welcome message
        st.session_state.chat_history.append({
            "user": "",
            "bot": "Hey there! I'm here to help with our awesome products. Ask about Paneer, Milk, or anything else!",
            "intent": "greeting",
            "product": None
        })
    else:
        st.session_state.chat_history.append({
            "user": "",
            "bot": "<div class='farewell-message'>! Thank you ! Have a good day !</div>",
            "intent": "farewell",
            "product": None
        })
    st.rerun()

with st.sidebar:
    with st.expander("Chat Window", expanded=st.session_state.show_chat):
        # CSS for farewell message
        st.markdown("""
            <style>
                .farewell-message {
                    background-color: #e6ffe6;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    color: #28a745;
                    font-weight: bold;
                }
            </style>
        """, unsafe_allow_html=True)

        user_input = st.text_input(
            "Ask about our products...",
            key=f"chat_input_{st.session_state.chat_input_counter}",
            value=""
        )

        if user_input and user_input != st.session_state.last_user_input:
            st.session_state.last_user_input = user_input
            response, intent, product = process_query(
                user_input, st.session_state.stock, base_prices, units, product_descriptions, client
            )
            if response:
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": response,
                    "intent": intent,
                    "product": product
                })
                st.session_state.chat_input_counter += 1
                st.session_state.last_user_input = None
                st.rerun()

        if st.session_state.chat_history:
            st.write("**Chat History**")
            for chat in st.session_state.chat_history[-5:]:
                st.markdown(f"**You**: {chat['user']}")
                st.markdown(f"**Assistant**: {chat['bot']}", unsafe_allow_html=True)
                st.markdown("---")

        # Close Chat button
        if st.button("Close Chat", key=f"close_chat_{st.session_state.update_trigger}"):
            st.session_state.show_chat = False
            st.session_state.chat_history.append({
                "user": "",
                "bot": "<div class='farewell-message'>! Thank you ! Have a good day !</div>",
                "intent": "farewell",
                "product": None
            })
            st.rerun()

# --- SIMULATION CONTROL ---
st.sidebar.header("📆 Start your day!")
if st.sidebar.button("Start Day"):
    st.session_state.day_started = True
    st.success(f"✅ Simulation Day Started: {st.session_state.simulation_day.strftime('%Y-%m-%d')}")

if st.sidebar.button("End Day"):
    ended_day = st.session_state.simulation_day
    st.session_state.simulation_day += timedelta(days=1)
    st.session_state.order_count = 1
    st.session_state.day_started = False
    st.session_state.checked_rows = set()
    st.session_state.sent_reminders = {}
    st.session_state.order_items = []
    st.success(
        f"✅ Day Ended: {ended_day.strftime('%Y-%m-%d')}. New day: {st.session_state.simulation_day.strftime('%Y-%m-%d')}"
    )

# Only show main functionality if day is started
if st.session_state.day_started:
    # --- CUSTOMER ORDERING ---
    st.subheader("🛍️ Place Your Order")
    product = st.selectbox("Select Product", st.session_state.stock['Product'])
    qty = st.number_input(f"Enter quantity for {product} ({units[product]})", min_value=1, step=1)

    if st.button("Add to Order"):
        available = st.session_state.stock.loc[st.session_state.stock['Product'] == product, 'Available_Stock'].values[0]
        if qty > available:
            st.error(f"Only {available} {units[product]} of {product} available.")
        else:
            st.session_state.order_items.append({'Product': product, 'Quantity': qty})
            st.success(f"Added {qty} {units[product]} of {product} to order.")

    if st.session_state.order_items:
        st.subheader("🧾 Current Order")
        st.dataframe(pd.DataFrame(st.session_state.order_items))

        if st.button("Done"):
            order_id = f"Order-{st.session_state.order_count}"
            for item in st.session_state.order_items:
                idx = st.session_state.stock[
                    st.session_state.stock['Product'] == item['Product']].index[0]
                st.session_state.stock.at[idx, 'Available_Stock'] -= item['Quantity']
                item['Order ID'] = order_id
            st.session_state.placed_orders.append({
                'Order ID': order_id,
                'Items': st.session_state.order_items,
                'Date': st.session_state.simulation_day.strftime('%Y-%m-%d')
            })
            st.session_state.order_items = []
            st.session_state.order_count += 1
            st.success("Order placed and stock updated.")

    # --- INVENTORY STATUS ---
    st.subheader(f"📦 Updated Inventory Schedule as on {st.session_state.simulation_day.strftime('%Y-%m-%d')}")

    def get_stock_status(row):
        if row['Available_Stock'] < row['ROL']:
            return 'Below Reorder Level'
        elif row['Available_Stock'] == row['ROL']:
            return 'At Reorder Level'
        return 'Sufficient Stock'

    def calculate_to_be_ordered(row):
        if row['Available_Stock'] <= row['ROL']:
            return row['Max_Stock'] - row['Available_Stock']
        return 'N/A'

    def calculate_time_to_stock_out(row):
        if row['Available_Stock'] <= row['ROL']:
            return f"{row['Available_Stock'] / row['Demand_Rate_per_Day']:.1f}"
        return 'N/A'

    def color_row(row):
        color = ''
        if row['Stock_Status'] == 'Below Reorder Level':
            color = 'background-color: red; color: white; animation: blink 1s infinite;'
        elif row['Stock_Status'] == 'At Reorder Level':
            color = 'background-color: orange; animation: blink 1s infinite;'
        return [color] * len(row)

    st.markdown("""<style>
        @keyframes blink {0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; }}
    </style>""", unsafe_allow_html=True)

    st.session_state.stock['Stock_Status'] = st.session_state.stock.apply(get_stock_status, axis=1)
    st.session_state.stock['To_be_Ordered'] = st.session_state.stock.apply(calculate_to_be_ordered, axis=1)
    st.session_state.stock['Time_to_Stock_Out'] = st.session_state.stock.apply(calculate_time_to_stock_out, axis=1)

    styled = st.session_state.stock.style.apply(color_row, axis=1)
    st.markdown(styled.to_html(), unsafe_allow_html=True)

    # --- SUPPLIER QUOTES ---
    show_supplier_quotes = False
    for row in st.session_state.stock.itertuples():
        if (row.Stock_Status in ['Below Reorder Level', 'At Reorder Level'] and
                not st.session_state.order_placed.get(row.Product, False) and
                not isinstance(row.To_be_Ordered, str)):
            show_supplier_quotes = True
            break

    if show_supplier_quotes:
        st.subheader("💬 Supplier Quotes (Top 5 Cheapest per Product)")
        for row in st.session_state.stock.itertuples():
            if row.Stock_Status in ['Below Reorder Level', 'At Reorder Level']:
                product = row.Product
                if st.session_state.order_placed.get(product, False):
                    continue
                to_order = row.To_be_Ordered
                if isinstance(to_order, str):
                    continue
                if product not in st.session_state.supplier_quotes:
                    mean_price = base_prices[product]
                    quotes = []
                    for i in range(5):
                        supplier = f"Supplier {chr(65 + i)}"
                        rate = round(np.random.normal(mean_price, mean_price * 0.1), 2)
                        promised_days = max(1, int(round(np.random.normal(row.Lead_Time_days, row.Lead_Time_days * 0.1))))
                        quotes.append({
                            'Supplier': supplier,
                            'Rate per Unit (Rs.)': f"₹{rate:.2f}",
                            'To be Ordered': to_order,
                            'Total Cost (Rs.)': rate * to_order,
                            'Promised Days': promised_days
                        })
                    st.session_state.supplier_quotes[product] = sorted(quotes, key=lambda x: x['Total Cost (Rs.)'])
                df = pd.DataFrame(st.session_state.supplier_quotes[product])
                selection = st.radio(f"Choose Supplier for {product}", df['Supplier'], key=f"select_{product}")
                st.dataframe(df)
                if st.button(f"Done for {product}", key=f"done_{product}"):
                    quote = df[df['Supplier'] == selection].iloc[0]
                    rate = float(quote['Rate per Unit (Rs.)'].strip('₹'))
                    promised_days = quote['Promised Days']
                    lead_time_days = int(st.session_state.stock.loc[
                                             st.session_state.stock['Product'] == product, 'Lead_Time_days'].values[0])
                    st.session_state.procurement_orders.append({
                        'Product': product,
                        'Supplier': selection,
                        'Units Ordered': to_order,
                        'Rate per Unit': f"₹{rate:.2f}",
                        'Total Cost': f"₹{rate * to_order:.2f}",
                        'Order_Date': st.session_state.simulation_day,
                        'Promised Days': promised_days,
                        'Promised_Date': st.session_state.simulation_day + timedelta(days=int(promised_days)),
                        'Lead_Time_days': lead_time_days,
                        'Lead_Time_Date': st.session_state.simulation_day + timedelta(days=lead_time_days),
                        'Order_Placed': "Yes",
                        'Received': False
                    })
                    st.session_state.order_placed[product] = True
                    st.success(f"{to_order} {units[product]} of {product} ordered from {selection}.")
                    st.rerun()

    # --- PROCUREMENT TRACKING ---
    if st.session_state.procurement_orders:
        st.subheader("🚚 Procurement Order Tracking")
        df_proc = pd.DataFrame(st.session_state.procurement_orders)
        df_proc['Received'] = df_proc['Received'].astype(bool)
        df_proc['Days Left'] = df_proc.apply(
            lambda x: max(0, x['Promised Days'] - (st.session_state.simulation_day - x['Order_Date']).days), axis=1)
        df_proc['Status'] = df_proc.apply(
            lambda x: "Overdue" if not x['Received'] and (st.session_state.simulation_day - x['Order_Date']).days > x[
                'Promised Days'] else "", axis=1)

        # CSS for table and warning styling
        st.markdown("""
        <style>
            @keyframes blink {0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; }}
            .stDataEditor { border-collapse: collapse; width: 100%; }
            .stDataEditor th, .stDataEditor td { border: 1px solid black; padding: 8px; text-align: left; }
            .stDataEditor th { background-color: #f2f2f2; }
            .overdue { background-color: #ffcccc; color: red; animation: blink 1s infinite; }
            .warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; margin: 10px 0; }
        </style>
        """, unsafe_allow_html=True)

        # Warning for orders due today with automatic email
        due_today_orders = df_proc[(df_proc['Received'] == False) &
                                   (df_proc.apply(
                                       lambda x: (st.session_state.simulation_day - x['Order_Date']).days == x[
                                           'Promised Days'], axis=1))]
        if not due_today_orders.empty:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ OVERDUE ALERT!</strong> The following orders are to be delivered today:
                <ul>
            """ + "\n".join([
                                f"<li>{row['Product']} from {row['Supplier']} (Ordered: {row['Order_Date'].strftime('%Y-%m-%d')}, Promised: {row['Promised_Date'].strftime('%Y-%m-%d')})</li>"
                                for _, row in due_today_orders.iterrows()]) + """
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Automatically send reminder emails
            for idx, row in due_today_orders.iterrows():
                order_key = f"{idx}_{st.session_state.simulation_day.strftime('%Y-%m-%d')}"
                if order_key not in st.session_state.sent_reminders:
                    # Send email
                    msg = EmailMessage()
                    msg.set_content(
                        f"Dear {row['Supplier']},\n\n"
                        f"This is a reminder that your order of {row['Units Ordered']} {units[row['Product']]} of {row['Product']}, "
                        f"placed on {row['Order_Date'].strftime('%Y-%m-%d')}, is due for delivery today, "
                        f"{row['Promised_Date'].strftime('%Y-%m-%d')}. Please ensure timely delivery.\n\n"
                        f"Order Details:\n"
                        f"- Product: {row['Product']}\n"
                        f"- Supplier: {row['Supplier']}\n"
                        f"- Units Ordered: {row['Units Ordered']} {units[row['Product']]}\n"
                        f"- Order Date: {row['Order_Date'].strftime('%Y-%m-%d')}\n"
                        f"- Promised Delivery Date: {row['Promised_Date'].strftime('%Y-%m-%d')}\n"
                        f"- Total Cost: {row['Total Cost']}\n\n"
                        f"Regards,\nBuyer"
                    )
                    msg['Subject'] = f"Reminder for the {row['Product']} order placed on {row['Order_Date'].strftime('%Y-%m-%d')}"
                    msg['From'] = "supplier123.sample@gmail.com"
                    msg['To'] = "supplier123.sample@gmail.com"

                    try:
                        with smtplib.SMTP("smtp.gmail.com", 587) as server:
                            server.starttls()
                            server.login("supplier123.sample@gmail.com", "vgdt fwsr yffb dbfr")
                            server.send_message(msg)
                        st.success(f"Reminder email sent for {row['Product']} from {row['Supplier']}!")
                        st.session_state.sent_reminders[order_key] = True
                    except Exception as e:
                        st.error(f"Failed to send reminder email for {row['Product']} from {row['Supplier']}: {str(e)}")

        # Warning for overdue orders
        overdue_orders = df_proc[(df_proc['Received'] == False) &
                                 (df_proc['Status'] == "Overdue")]
        if not due_today_orders.empty:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ OVERDUE ALERT!</strong> The following orders are overdue:
                <ul>
            """ + "\n".join([
                                f"<li>{row['Product']} from {row['Supplier']} (Ordered: {row['Order_Date'].strftime('%Y-%m-%d')}, Promised: {row['Promised_Date'].strftime('%Y-%m-%d')})</li>"
                                for _, row in overdue_orders.iterrows()]) + """
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Apply styling to overdue rows
        def style_procurement_row(row):
            styles = [''] * len(row)
            if row['Status'] == "Overdue" and not row['Received']:
                styles = ['background-color: #ffcccc; color: red; animation: blink 1s infinite;'] * len(row)
            return styles

        # Render table with data_editor
        if not df_proc.empty:
            disabled_columns = ["Product", "Supplier", "Units Ordered", "Rate per Unit", "Total Cost",
                                "Order_Date", "Promised Days", "Promised_Date", "Lead_Time_days",
                                "Lead_Time_Date", "Order_Placed", "Days Left", "Status"]
            edited_df = st.data_editor(
                df_proc.style.apply(style_procurement_row, axis=1),
                column_config={
                    "Product": st.column_config.TextColumn(width="medium"),
                    "Supplier": st.column_config.TextColumn(width="medium"),
                    "Units Ordered": st.column_config.NumberColumn(width="small"),
                    "Rate per Unit": st.column_config.TextColumn(width="small"),
                    "Total Cost": st.column_config.TextColumn(width="small"),
                    "Order_Date": st.column_config.DateColumn("Order Date", format="YYYY-MM-DD", width="medium"),
                    "Promised Days": st.column_config.NumberColumn(width="small"),
                    "Promised_Date": st.column_config.DateColumn("Promised Date", format="YYYY-MM-DD", width="medium"),
                    "Lead_Time_days": st.column_config.NumberColumn("Lead Time Days", width="small"),
                    "Lead_Time_Date": st.column_config.DateColumn("Lead Time Date", format="YYYY-MM-DD",
                                                                  width="medium"),
                    "Order_Placed": st.column_config.TextColumn(width="small"),
                    "Days Left": st.column_config.NumberColumn(width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Received": st.column_config.CheckboxColumn(
                        "Received",
                        help="Check to mark as received",
                        default=False
                    )
                },
                disabled=disabled_columns,
                hide_index=True,
                key=f"procurement_editor_{st.session_state.update_trigger}"
            )

            for idx, row in edited_df.iterrows():
                original_order = st.session_state.procurement_orders[idx]
                if row['Received'] and not original_order['Received'] and idx not in st.session_state.checked_rows:
                    st.session_state.checked_rows.add(idx)
                    st.session_state.pending_confirmation = {
                        'index': idx,
                        'product': row['Product'],
                        'supplier': row['Supplier'],
                        'units_ordered': row['Units Ordered'],
                        'total_cost': row['Total Cost'],
                        'order_date': row['Order_Date']
                    }

        if st.session_state.pending_confirmation:
            pending = st.session_state.pending_confirmation
            st.warning(
                f"Can you confirm that the order from {pending['supplier']} for {pending['units_ordered']} "
                f"{units[pending['product']]} of {pending['product']} at {pending['total_cost']} "
                f"ordered on {pending['order_date']} has been received?"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("OK", key=f"confirm_{pending['index']}"):
                    idx = pending['index']
                    product = pending['product']
                    units_ordered = pending['units_ordered']
                    idx_stock = st.session_state.stock[st.session_state.stock['Product'] == product].index[0]
                    st.session_state.stock.at[idx_stock, 'Available_Stock'] += units_ordered
                    max_stock = st.session_state.stock.at[idx_stock, 'Max_Stock']
                    st.session_state.stock.at[idx_stock, 'Available_Stock'] = min(
                        st.session_state.stock.at[idx_stock, 'Available_Stock'], max_stock)
                    lead_time = st.session_state.stock.loc[
                        st.session_state.stock['Product'] == product, 'Lead_Time_days'].values[0]
                    promised_days = st.session_state.procurement_orders[idx]['Promised Days']
                    delay_days = (st.session_state.simulation_day - (
                            pending['order_date'] + timedelta(days=int(promised_days)))).days
                    st.session_state.procurement_orders[idx]['Received'] = True
                    st.session_state.procurement_orders[idx]['Days_Delay'] = max(0, delay_days)
                    st.session_state.order_placed[product] = False
                    st.session_state.pending_confirmation = None
                    st.session_state.update_trigger += 1
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_{pending['index']}"):
                    st.session_state.checked_rows.discard(pending['index'])
                    st.session_state.pending_confirmation = None
                    st.rerun()

    # --- CUSTOMER ORDER SUMMARY ---
    if st.session_state.placed_orders:
        st.subheader("🧑‍🤝‍🧑 Customers' Order Summary")
        all_orders = []
        for order in st.session_state.placed_orders:
            for item in order['Items']:
                all_orders.append({
                    'Order ID': order['Order ID'],
                    'Date': order['Date'],
                    'Product': item['Product'],
                    'Quantity': item['Quantity'],
                    'Current Stock': st.session_state.stock[
                        st.session_state.stock['Product'] == item['Product']]['Available_Stock'].values[0]
                })
        st.dataframe(pd.DataFrame(all_orders))

    # --- SUPPLIER PERFORMANCE REPORT ---
    if st.sidebar.button("Show Supplier Performance Report"):
        st.subheader("📊 Supplier Performance Report (Last 10 Days)")
        st.session_state.supplier_performance = {}
        min_date = st.session_state.simulation_day - timedelta(days=9)

        for order in st.session_state.procurement_orders:
            if order['Order_Date'] < min_date:
                continue
            supplier = order['Supplier']
            product = order['Product']
            delay_days = order.get('Days_Delay', 0)
            price = float(order['Rate per Unit'].strip('₹'))
            base_price = base_prices.get(product, 0)

            if supplier not in st.session_state.supplier_performance:
                st.session_state.supplier_performance[supplier] = {}

            if product not in st.session_state.supplier_performance[supplier]:
                st.session_state.supplier_performance[supplier][product] = {
                    'total_delay': 0,
                    'total_price': 0,
                    'count': 0,
                    'base_price': base_price
                }

            st.session_state.supplier_performance[supplier][product]['total_delay'] += delay_days
            st.session_state.supplier_performance[supplier][product]['total_price'] += price
            st.session_state.supplier_performance[supplier][product]['count'] += 1

        supplier_performance = {
            'Supplier': [],
            'Product': [],
            'Orders': [],
            'Average Delay (Days)': [],
            'Average Price (Rs.)': [],
            'Base Price (Rs.)': []
        }

        for supplier, products in st.session_state.supplier_performance.items():
            for product, metrics in products.items():
                supplier_performance['Supplier'].append(supplier)
                supplier_performance['Product'].append(product)
                supplier_performance['Orders'].append(metrics['count'])
                supplier_performance['Average Delay (Days)'].append(
                    round(metrics['total_delay'] / metrics['count'], 2) if metrics['count'] > 0 else 0)
                supplier_performance['Average Price (Rs.)'].append(
                    round(metrics['total_price'] / metrics['count'], 2) if metrics['count'] > 0 else 0)
                supplier_performance['Base Price (Rs.)'].append(metrics['base_price'])

        performance_df = pd.DataFrame(supplier_performance)
        if not performance_df.empty:
            st.dataframe(performance_df)
        else:
            st.info("No procurement orders in the last 10 days.")