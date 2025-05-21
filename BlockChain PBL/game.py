import streamlit as st
import hashlib
import secrets
import time
import json

class Wallet:
    def __init__(self):
        self.private_key = secrets.token_hex(32)
        self.public_key = self.generate_public_key(self.private_key)
        self.address = self.generate_address(self.public_key)

    def generate_public_key(self, private_key):
        return hashlib.sha256(private_key.encode()).hexdigest()

    def generate_address(self, public_key):
        return hashlib.sha256(public_key.encode()).hexdigest()

    def __str__(self):
        return f"Wallet Address: {self.address}\nPublic Key: {self.public_key[:10]}...\nPrivate Key (Simplified): {self.private_key[:10]}... (Keep this secret!)"


class Block:
    def __init__(self, index, timestamp, transactions, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = str(self.index) + str(self.timestamp) + str(self.transactions) + str(self.previous_hash) + str(self.nonce)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __str__(self, wallet_address=None):
        block_info =  f"**Block #{self.index}**\n" \
                      f"* Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n"

        if self.transactions:
            block_info += "* Transactions:\n"
            for tx in self.transactions:
                sender = tx['sender']
                recipient = tx['recipient']
                amount = tx['amount']

                sender_display = "System Reward" if sender is None else f"`{sender[:8]}...`"
                recipient_display = f"`{recipient[:8]}...`"

                if sender is None:
                    recipient_display = "**Miner Reward**"
                elif sender == wallet_address:
                    sender_display = "**You**"
                elif recipient == wallet_address:
                    recipient_display = "**You**"
                elif recipient == "genesis_miner":
                    recipient_display = "Genesis Distribution"

                block_info += f"    - {sender_display} sent {amount} coins to {recipient_display}\n"
        else:
            block_info += "* Transactions: No transactions in this block\n"

        block_info += f"* Previous Block Code: `{self.previous_hash[:8]}...`\n"
        block_info += f"* Block Code: `{self.hash[:8]}...`\n"
        return block_info


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.difficulty = 1
        self.mining_reward = 10

    def create_genesis_block(self):
        return Block(0, time.time(), [], "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, sender_address, recipient_address, amount):
        transaction = {
            'sender': sender_address,
            'recipient': recipient_address,
            'amount': amount
        }
        self.pending_transactions.append(transaction)
        return True

    def mine_block(self, miner_address):
        if not self.pending_transactions:
            return None

        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions,
            previous_hash=self.get_latest_block().hash
        )
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

        reward_transaction = {
            'sender': None,
            'recipient': miner_address,
            'amount': self.mining_reward
        }
        self.pending_transactions = [reward_transaction]

        return new_block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                st.sidebar.error(f"Block {current_block.index} hash is invalid.")
                return False

            if current_block.previous_hash != previous_block.hash:
                st.sidebar.error(f"Block {current_block.index} previous hash is invalid.")
                return False
        return True

    def get_balance(self, address):
        balance = 0
        for block in self.chain:
            for transaction in block.transactions:
                if transaction['sender'] == address:
                    balance -= transaction['amount']
                if transaction['recipient'] == address:
                    balance += transaction['amount']
        return balance

    def print_chain(self):
        chain_str = ""
        for block in self.chain:
            chain_str += str(block) + "\n"
        return chain_str



st.set_page_config(page_title="Simple Blockchain Wallet", page_icon="🪙")

with st.sidebar:
    st.header("About This Wallet")
    st.write("Welcome to the **Simple Python Blockchain Wallet**! This is an educational project demonstrating the basic concepts of blockchain technology in a fun and easy way.")
    st.markdown("---")
    st.subheader("How to Use")
    st.write("**1. View Balance:** See how many coins you have in your wallet.")
    st.write("**2. Send Coins:** Transfer coins to another wallet address. Think of it like sending a digital gift!")
    st.write("**3. Mine Block:**  Like adding a page to a digital ledger. This groups pending transactions into a block and adds it to the blockchain. You get a small reward (**Miner Reward**) for 'mining'!")
    st.write("**4. View Blockchain:** Explore the digital ledger (the blockchain) and see all the transactions recorded.")
    st.markdown("---")
    st.subheader("Blockchain Basics (Simplified)")
    st.write("**Imagine a digital notebook (the blockchain):**")
    st.write("* **Blocks:** Each page in the notebook is a 'block'. It contains a list of recent 'transactions'.")
    st.write("* **Transactions:**  Records of who sent coins to whom and how much.")
    st.write("* **Chain:** Pages are linked together in a 'chain' using special codes (hashes). This makes it secure and tamper-proof.")
    st.write("* **Mining:**  'Mining' is like adding a new page to the notebook. In our simplified version, it's just adding pending transactions to the blockchain. You get a **Miner Reward** for each block you mine!")
    st.markdown("---")
    st.subheader("Important Notes")
    st.write("* **Educational Only:** This is a simplified blockchain for learning. It's not real cryptocurrency and not secure for real use.")
    st.write("* **Simplified Keys:**  Keys and addresses are basic for demonstration.")
    st.write("* **Simplified Mining:** Mining is automatic and doesn't involve real 'proof-of-work'.")
    st.write("* **Single User:** This blockchain is just for you to play with.")
    st.markdown("---")
    st.info("Have fun exploring blockchain!")


st.title("🪙 Simple Python Blockchain Wallet")
st.markdown("---")

if 'blockchain' not in st.session_state:
    st.session_state.blockchain = Blockchain()
if 'wallet' not in st.session_state:
    st.session_state.wallet = Wallet()

wallet = st.session_state.wallet
blockchain = st.session_state.blockchain

st.subheader("Your Wallet Information")
st.info(f"**Wallet Address:** `{wallet.address}`\n\n**This is your public address - like your account number. Share it to receive coins.** (Private and Public Keys are simplified for this demo)")

st.markdown("---")

action = st.selectbox("Choose an Action:",
                      ["View Balance", "Send Coins", "Mine Block", "View Blockchain"],
                      help="Select what you want to do with your wallet.")

st.markdown("---")

if action == "View Balance":
    st.subheader("💰 Check Your Wallet Balance")
    balance = blockchain.get_balance(wallet.address)
    st.metric(label="Current Balance", value=f"{balance} Coins", help="This is the total number of coins in your wallet.")

elif action == "Send Coins":
    st.subheader("💸 Send Coins")
    recipient_address = st.text_input("Recipient Wallet Address:", placeholder="Enter recipient's address", help="Enter the address of the wallet you want to send coins to.")
    amount = st.number_input("Amount to Send:", min_value=0.0, format="%.2f", help="Enter the amount of coins you want to send.")
    if st.button("Send Coins", help="Click here to send the coins."):
        if amount <= 0:
            st.error("Amount must be positive. You can't send zero or negative coins!")
        elif blockchain.get_balance(wallet.address) < amount:
            st.error("Oops! Not enough coins in your wallet. Check your balance.")
        elif blockchain.add_transaction(wallet.address, recipient_address, amount):
            st.success(f"✅ Transaction of {amount} coins to `{recipient_address[:8]}...` added to pending transactions! It will be processed when the next block is mined.")
        else:
            st.error("❌  Something went wrong adding the transaction. Please try again.")

elif action == "Mine Block":
    st.subheader("⛏️ Mine Block and Add Transactions")
    st.write("Mining adds all pending transactions to the blockchain in a new block. You also get a **Miner Reward** for mining!")
    miner_address = wallet.address
    if st.button("Mine Block Now", help="Click to mine a new block and process pending transactions."):
        if not blockchain.pending_transactions:
            st.info("ℹ️ No pending transactions to mine right now. Send some coins first, then come back to mine!")
        else:
            with st.spinner("Mining block... (Simplified process)"):
                new_block = blockchain.mine_block(miner_address)
                time.sleep(1)
            if new_block:
                st.success(f"🎉 Block #{new_block.index} mined successfully! Block Code: `{new_block.hash[:8]}...`\nTransactions added to the blockchain. You also received a **Miner Reward**!")
            else:
                st.error("❌ Mining failed. Please try again.")

elif action == "View Blockchain":
    st.subheader("⛓️ Blockchain Explorer")
    st.write("Here's the current blockchain, showing all the blocks and transactions recorded so far:")

    for block in blockchain.chain:
        with st.expander(f"Block #{block.index} - Block Code: {block.hash[:10]}...", expanded=(block.index == blockchain.get_latest_block().index)):
            st.write(block.__str__(wallet.address))
        st.markdown("---")

    if blockchain.pending_transactions:
        st.subheader("⏳ Pending Transactions (Not yet in a block)")
        st.write("These transactions are waiting to be added to the blockchain in the next mined block:")
        for tx in blockchain.pending_transactions:
            sender = tx['sender']
            recipient = tx['recipient']
            amount = tx['amount']

            sender_display = "System Reward" if sender is None else f"`{sender[:8]}...`"
            recipient_display = f"`{recipient[:8]}...`"

            if sender == wallet.address:
                sender_display = "**You**"
            if recipient == wallet.address:
                recipient_display = "**You**"
            elif recipient == "genesis_miner":
                recipient_display = "Genesis Distribution"

            st.write(f"- Pending Transaction: {sender_display} to {recipient_display} - Amount: {amount} coins")

    else:
        st.info("No pending transactions.")

if len(blockchain.chain) == 1 and st.session_state.get('initial_coins_distributed', False) is False:
    blockchain.add_transaction(None, wallet.address, 100)
    blockchain.mine_block("genesis_miner")
    st.session_state['initial_coins_distributed'] = True
    st.success("🎁 Initial 100 coins distributed to your wallet! Start exploring!")