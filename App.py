import streamlit as st
import pandas as pd
import numpy as np

# Define base prices
base_prices = {
    'Paneer': 250,
    'Milk': 70,
    'Onions': 60,
    'Potatoes': 55,
    'Meat': 320,
    'Oil': 100
}

# Product data
products_data = {
    'Product': ['Paneer', 'Milk', 'Onions', 'Potatoes', 'Meat', 'Oil'],
    'Opening_Stock': [100, 200, 150, 180, 120, 250],
    'Minimum_Level': [20, 30, 40, 50, 30, 70],
    'Lead_Time_days': [3, 2, 4, 5, 3, 7],
    'Max_Stock': [100, 200, 150, 180, 120, 250]
}

# Demand rate
demand_rate = {
    'Paneer': 10,
    'Milk': 15,
    'Onions': 8,
    'Potatoes': 12,
    'Meat': 5,
    'Oil': 20
}

units = {
    'Paneer': 'kgs', 'Milk': 'packets', 'Onions': 'kgs',
    'Potatoes': 'kgs', 'Meat': 'kgs', 'Oil': 'kgs'
}

# Initialize stock DataFrame
inventory_df = pd.DataFrame(products_data)
inventory_df['Demand_Rate_per_Day'] = inventory_df['Product'].map(demand_rate)
inventory_df['ROL'] = inventory_df.apply(
    lambda row: (row['Demand_Rate_per_Day'] * row['Lead_Time_days']) + row['Minimum_Level'], axis=1
)
inventory_df['Units'] = inventory_df['Product'].map(units)

# Initialize session state
for key, value in {
    'stock': inventory_df.copy(),
    'order_items': [],
    'placed_orders': [],
    'order_count': 1,
    'supplier_quotes': {},
    'procurement_orders': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# UI Title
st.title("🛒 Product Ordering App")

# Product selection
product = st.selectbox("Select Product", st.session_state.stock['Product'])
qty = st.number_input(f"Enter quantity for {product} ({units[product]})", min_value=1, step=1)

if st.button("Add to Order"):
    available = st.session_state.stock.loc[
        st.session_state.stock['Product'] == product, 'Opening_Stock'].values[0]
    if qty > available:
        st.error(f"Only {available} {units[product]} of {product} available.")
    else:
        st.session_state.order_items.append({'Product': product, 'Quantity': qty})
        st.success(f"Added {qty} {units[product]} of {product} to order.")

# Show current order
if st.session_state.order_items:
    st.subheader("🧾 Current Order")
    st.dataframe(pd.DataFrame(st.session_state.order_items))

    if st.button("Done"):
        order_id = f"Order-{st.session_state.order_count}"
        for item in st.session_state.order_items:
            idx = st.session_state.stock[
                st.session_state.stock['Product'] == item['Product']].index[0]
            st.session_state.stock.at[idx, 'Opening_Stock'] -= item['Quantity']
        for item in st.session_state.order_items:
            item['Order ID'] = order_id
        st.session_state.placed_orders.append({
            'Order ID': order_id,
            'Items': st.session_state.order_items
        })
        st.session_state.order_items = []
        st.session_state.order_count += 1
        st.success("Order placed and stock updated.")

# Update Inventory Status
st.subheader("📦 Updated Inventory")

def get_stock_status(row):
    if row['Opening_Stock'] < row['ROL']:
        return 'Below Reorder Level'
    elif row['Opening_Stock'] == row['ROL']:
        return 'At Reorder Level'
    return 'Sufficient Stock'

def calculate_to_be_ordered(row):
    return row['Max_Stock'] - row['Opening_Stock'] if row['Opening_Stock'] <= row['ROL'] else 'N/A'

def calculate_time_to_stock_out(row):
    return f"{row['Opening_Stock'] / row['Demand_Rate_per_Day']:.1f}" if row['Opening_Stock'] <= row['ROL'] else 'N/A'

def color_row(row):
    color = ''
    if row['Stock_Status'] == 'Below Reorder Level':
        color = 'background-color: red; color: white; animation: blink 1s infinite;'
    elif row['Stock_Status'] == 'At Reorder Level':
        color = 'background-color: orange; animation: blink 1s infinite;'
    return [color] * len(row)

st.markdown("""
    <style>
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

st.session_state.stock['Stock_Status'] = st.session_state.stock.apply(get_stock_status, axis=1)
st.session_state.stock['To_be_Ordered'] = st.session_state.stock.apply(calculate_to_be_ordered, axis=1)
st.session_state.stock['Time_to_Stock_Out'] = st.session_state.stock.apply(calculate_time_to_stock_out, axis=1)

styled = st.session_state.stock.style.apply(color_row, axis=1)
st.markdown(styled.to_html(), unsafe_allow_html=True)

# Supplier Quotes
st.subheader("💬 Supplier Quotes (Top 3 Cheapest per Product)")

for row in st.session_state.stock.itertuples():
    if row.Stock_Status in ['Below Reorder Level', 'At Reorder Level']:
        product = row.Product
        to_order = row.To_be_Ordered
        if to_order == 'N/A': continue

        if product not in st.session_state.supplier_quotes:
            mean_price = base_prices[product]
            quotes = []
            for i in range(5):
                supplier = f"Supplier {chr(65+i)}"
                rate = round(np.random.normal(mean_price, mean_price*0.1), 2)
                quotes.append({
                    'Supplier': supplier,
                    'Rate per Unit (Rs.)': f"₹{rate:.2f}",
                    'To be Ordered': to_order,
                    'Total Cost (Rs.)': rate * to_order
                })
            st.session_state.supplier_quotes[product] = sorted(quotes, key=lambda x: x['Total Cost (Rs.)'])

        df = pd.DataFrame(st.session_state.supplier_quotes[product])
        selection = st.radio(f"Choose Supplier for {product}", df['Supplier'], key=product)
        st.dataframe(df)

        if st.button(f"Done for {product}"):
            quote = df[df['Supplier'] == selection].iloc[0]
            rate = float(quote['Rate per Unit (Rs.)'].strip('₹'))
            total = to_order
            st.session_state.stock.loc[
                st.session_state.stock['Product'] == product, 'Opening_Stock'] += total
            st.session_state.procurement_orders.append({
                'Product': product,
                'Supplier': selection,
                'Units Ordered': total,
                'Rate per Unit': f"₹{rate:.2f}",
                'Total Cost': f"₹{rate * total:.2f}"
            })
            st.success(f"{total} {units[product]} of {product} ordered from {selection}.")

# Customers' Order Summary
if st.session_state.placed_orders:
    st.subheader("🧑‍🤝‍🧑 Customers' Order Summary")
    all_orders = []
    for order in st.session_state.placed_orders:
        for item in order['Items']:
            all_orders.append({
                'Order ID': order['Order ID'],
                'Product': item['Product'],
                'Quantity': item['Quantity'],
                'Current Stock': st.session_state.stock[
                    st.session_state.stock['Product'] == item['Product']]['Opening_Stock'].values[0]
            })
    st.dataframe(pd.DataFrame(all_orders))

# Procurement Order Summary (Persistent)
if st.session_state.procurement_orders:
    st.subheader("📦 Procurement Order Summary")
    st.dataframe(pd.DataFrame(st.session_state.procurement_orders))
