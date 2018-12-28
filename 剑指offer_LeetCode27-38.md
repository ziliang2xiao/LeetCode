### 剑指offer——LeetCode~27-38

##### 4 解决面试题的思路

###### 4.2 画图让抽象问题形象化

> 面试题27：二叉树的镜像

1. [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree) (226)

   Invert a binary tree.

   **Example:**

   Input:

   ```
        4
      /   \
     2     7
    / \   / \
   1   3 6   9
   ```

   Output:

   ```
        4
      /   \
     7     2
    / \   / \
   9   6 3   1
   ```

   **Trivia:**
   This problem was inspired by [this original tweet](https://twitter.com/mxcl/status/608682016205344768) by [Max Howell](https://twitter.com/mxcl):

   > Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so f*** off.

   > Code_Python

   ```python
   # Definition for a binary tree node.
   # class TreeNode:
   #     def __init__(self, x):
   #         self.val = x
   #         self.left = None
   #         self.right = None
   
   # 非递归的做法
   class Solution:
       def invertTree(self, root):
           """
           :type root: TreeNode
           :rtype: TreeNode
           """
           if not root:return root
           tree_list = []
           tree_list.append(root)
           while len(tree_list)>0:
               root_sub = tree_list.pop()
               root_left = root_sub.left
               root_sub.left = root_sub.right
               root_sub.right = root_left
               if root_sub.left:tree_list.append(root_sub.left)
               if root_sub.right:tree_list.append(root_sub.right)
           return root
   ```

   > Code_Java
   
   ```java
   /**
    * Definition for a binary tree node.
    * public class TreeNode {
    *     int val;
    *     TreeNode left;
    *     TreeNode right;
    *     TreeNode(int x) { val = x; }
    * }
    */
   class Solution {
       public TreeNode invertTree(TreeNode root) {
           if(root==null){return null;}
           //创建临时的之前的左节点
           TreeNode preLeft = root.left;
           root.left = invertTree(root.right);
           root.right = invertTree(preLeft);
           return root
               
       }
   }
   //非递归做法请参考python部分
   
   more info: https://leetcode.com/problems/invert-binary-tree/discuss/62707/Straightforward-DFS-recursive-iterative-BFS-solutions
   ```

> 面试题28：对称的二叉树

1. [Symmetric Tree](https://leetcode.com/problems/symmetric-tree) (101)

   Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

   For example, this binary tree `[1,2,2,3,4,4,3]` is symmetric:

   ```
       1
      / \
     2   2
    / \ / \
   3  4 4  3
   ```

   But the following `[1,2,2,null,3,null,3]` is not:

   ```
       1
      / \
     2   2
      \   \
      3    3
   ```

   **Note:**
   Bonus points if you could solve it both recursively and iteratively.

   > Code_Python


   > Code_Java

   ```java
   /**
    * Definition for a binary tree node.
    * public class TreeNode {
    *     int val;
    *     TreeNode left;
    *     TreeNode right;
    *     TreeNode(int x) { val = x; }
    * }
    */
   class Solution {
       public boolean isSymmetric(TreeNode root) {
           //递归两条件，递归体，以及递归停止条件
           if (root==null){return true;}
           //将根节点分为左右子树进行判断
           return isSame(root.left,root.right);
       }
       public boolean isSame(TreeNode rootLeft,TreeNode rootRight){
           //递归停止条件，如果两个都为空，则为True
           if(rootLeft == null && rootRight == null){
               return true;
           }
           //递归False停止条件
           if(rootLeft == null || rootRight==null || rootLeft.val != rootRight.val){
               return false;
           }
           //分别递归左子树的左节点和右子树的右节点，以及左子树的右节点和右子树的左节点
           return isSame(rootLeft.left,rootRight.right) && isSame(rootLeft.right,rootRight.left);
       }
   }
   
   // more info: https://leetcode.com/problems/symmetric-tree/discuss/33054/Recursive-and-non-recursive-solutions-in-Java
   ```

> 面试题29：顺时针打印矩阵

1. [Spiral Matrix](https://leetcode.com/problems/spiral-matrix) (54)

   Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

   **Example 1:**

   ```
   Input:
   [
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
    [ 7, 8, 9 ]
   ]
   Output: [1,2,3,6,9,8,7,4,5]
   ```

   **Example 2:**

   ```
   Input:
   [
     [1, 2, 3, 4],
     [5, 6, 7, 8],
     [9,10,11,12]
   ]
   Output: [1,2,3,4,8,12,11,10,9,5,6,7] 
   ```

   > Code_Java

   ```java
   public class Solution {
       public List<Integer> spiralOrder(int[][] matrix) {
           List<Integer> res = new ArrayList<>();
           if(matrix == null || matrix.length == 0)
               return res;
           int rowNum = matrix.length, colNum = matrix[0].length;
           int left = 0, right = colNum - 1, top = 0, bot = rowNum - 1;
           while(res.size() < rowNum * colNum) {
               for(int col = left; col <= right; col++)
                   res.add(matrix[top][col]);
               top++;
               if(res.size() < rowNum * colNum) {
                   for(int row = top; row <= bot; row++)
                       res.add(matrix[row][right]);
                   right--;   
               }
               if(res.size() < rowNum * colNum) {
                   for(int col = right; col >= left; col--)
                       res.add(matrix[bot][col]);
                   bot--;
               }
               if(res.size() < rowNum * colNum) {
                   for(int row = bot; row >= top; row--)
                       res.add(matrix[row][left]);
                   left++;
               }
           } 
           return res;
       }
   }　　
   
   // more info https://leetcode.com/problems/spiral-matrix/discuss/20599/Super-Simple-and-Easy-to-Understand-Solution/20830
   ```

###### 4.3举例让抽象问题具体化

> 面试题30：包含min函数的栈

1. [Min Stack](https://leetcode.com/problems/min-stack) (155)

   > Description

   Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

   - push(x) -- Push element x onto stack.
   - pop() -- Removes the element on top of the stack.
   - top() -- Get the top element.
   - getMin() -- Retrieve the minimum element in the stack.

   **Example:**

   ```
   MinStack minStack = new MinStack();
   minStack.push(-2);
   minStack.push(0);
   minStack.push(-3);
   minStack.getMin();   --> Returns -3.
   minStack.pop();
   minStack.top();      --> Returns 0.
   minStack.getMin();   --> Returns -2.
   ```

   > Code_Java

   ```java
   class MinStack {
   
       Stack<Integer> stack;
       //存放最小值列表
       Stack<Integer> patchStack;
       /** initialize your data structure here. */
       public MinStack() {
           stack = new Stack<Integer>();
           patchStack = new Stack<Integer>();
       }
       
       public void push(int x) {
           stack.push(x);
           if(patchStack.empty() || patchStack.peek()>x){
               patchStack.push(x);
           }else{
               patchStack.push(patchStack.peek());
           }
       }
       
       public void pop() {
           patchStack.pop();
           stack.pop();
       }
       
       public int top() {
           return stack.peek();
       }
       
       public int getMin() {
           return patchStack.peek();
           
       }
   }
   
   // one-stack solution: https://leetcode.com/problems/min-stack/discuss/49031/Share-my-Java-solution-with-ONLY-ONE-stack 
   
   /**
    * Your MinStack object will be instantiated and called as such:
    * MinStack obj = new MinStack();
    * obj.push(x);
    * obj.pop();
    * int param_3 = obj.top();
    * int param_4 = obj.getMin();
    */
   ```

> 面试题31：栈的压入、弹出序列

1. [Validate Stack Sequences](https://leetcode.com/problems/validate-stack-sequences/) (946)

   > Description

   输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。

   > Code_Java

   ```java
   public class Solution {
       public boolean IsPopOrder(int [] pushA,int [] popA) {
           if(pushA.length == 0 || popA.length == 0)
               return false;
           Stack<Integer> s = new Stack<Integer>();
           //用于标识弹出序列的位置
           int popIndex = 0;
           for(int i = 0; i< pushA.length;i++){
               s.push(pushA[i]);
               //如果栈不为空，且栈顶元素等于弹出序列
               while(!s.empty() &&s.peek() == popA[popIndex]){
                   //出栈
                   s.pop();
                   //弹出序列向后一位
                   popIndex++;
               }
           }
           //如果最后辅助栈为空则匹配成功
           return s.empty();
       }
   }
   ```

> 面试题32：从上到下打印二叉树

1. LeetCode**无

   > Description

   从上往下打印出二叉树的每个节点，同层节点从左至右打印。 

   > Code_Java

   ```java
   /**
   public class TreeNode {
       int val = 0;
       TreeNode left = null;
       TreeNode right = null;
   
       public TreeNode(int val) {
           this.val = val;
   
       }
   
   }
   */
   public class Solution {
       public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
           ArrayList<Integer> resultList = new ArrayList<>();
           if(root==null){return resultList;}
           //用于暂时存放未遍历的节点
           Queue<TreeNode> queue = new LinkedList<TreeNode>();
           queue.offer(root);
           while(!queue.isEmpty()){
               TreeNode tempNode = queue.poll();
               resultList.add(tempNode.val);
               if(tempNode.left!=null){queue.offer(tempNode.left);}
               if(tempNode.right!=null){queue.offer(tempNode.right);}
           }
           return resultList;
       }
   }
   ```

> 面试题33：二叉搜索树的后序遍历序列

1. LeetCode**无

   > Description

   输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。 


   > Code_Java

   ```java
   public class Solution {
        public boolean isBst(int[] arr, int start, int root) {
           if (start >= root)
               return true;
           int index = start;
           // 二叉搜索树中左子树节点的值小于根节点的值
           while (arr[index] < arr[root] && index<root) {
               index++;
           }
           // 判断右子树是否有数字小于root节点的值
           for (int i = index ; i < root-1; i++) {
               if (arr[i] < arr[root]) {
                   return false;
               }
           }
           // 遍历左右子树
           return isBst(arr, start, index - 1) && isBst(arr, index, root - 1);
       }
        public boolean VerifySquenceOfBST(int [] sequence) {
           if(sequence.length==0){
               return false;
           }
           return isBst(sequence,0,sequence.length-1);
       }
   
   }
   ```

> 面试题34：二叉树中和为某一值的路径

1. [Path Sum](https://leetcode.com/problems/path-sum) (112)

   > Description

   Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

   **Note:** A leaf is a node with no children.

   **Example:**

   Given the below binary tree and `sum = 22`,

   ```
         5
        / \
       4   8
      /   / \
     11  13  4
    /  \      \
   7    2      1
   ```

   return true, as there exist a root-to-leaf path `5->4->11->2` which sum is 22.


   > Code_Java

   ```java
   /**
    * Definition for a binary tree node.
    * public class TreeNode {
    *     int val;
    *     TreeNode left;
    *     TreeNode right;
    *     TreeNode(int x) { val = x; }
    * }
    */
   class Solution {
       public boolean hasPathSum(TreeNode root, int sum) {
           if(root==null){return false;}
           if(root.left==null && root.right == null && root.val == sum){return true;}
           return hasPathSum(root.left,sum-root.val) || hasPathSum(root.right,sum-root.val);
       }
   }
   ```

2. [Path Sum II](https://leetcode.com/problems/path-sum-ii) (113)

   > Description

   Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

   **Note:** A leaf is a node with no children.

   **Example:**

   Given the below binary tree and `sum = 22`,

   ```
         5
        / \
       4   8
      /   / \
     11  13  4
    /  \    / \
   7    2  5   1
   ```

   Return:

   ```
   [
      [5,4,11,2],
      [5,8,4,5]
   ]
   ```

   > Code_Python

   ```python
   #非递归
   
   # Definition for a binary tree node.
   # class TreeNode(object):
   #     def __init__(self, x):
   #         self.val = x
   #         self.left = None
   #         self.right = None
   
   class Solution(object):
       def pathSum(self, root, sum):
           """
           :type root: TreeNode
           :type sum: int
           :rtype: List[List[int]]
           """
           if not root:return []
           res = []
           queue = [(root, sum, [root.val])]
           while queue:
               curr, val, ls = queue.pop(0)
               if not curr.left and not curr.right and val == curr.val:
                   res.append(ls)
               if curr.left:
                   queue.append((curr.left, val-curr.val, ls+[curr.left.val]))
               if curr.right:
                   queue.append((curr.right, val-curr.val, ls+[curr.right.val]))
           return res
   ```

   > Code_Java

   ```java
   /**
    * Definition for a binary tree node.
    * public class TreeNode {
    *     int val;
    *     TreeNode left;
    *     TreeNode right;
    *     TreeNode(int x) { val = x; }
    * }
    */
   class Solution {
       public List<List<Integer>> pathSum(TreeNode root, int sum) {
            List<List<Integer>> List=new ArrayList<List<Integer>>();
   		 List<Integer> sub=new ArrayList<Integer>();
   		 helperDFS(root,sum,List,sub);
   		 return List;
   	 }
   	 private void helperDFS(TreeNode root,int sum,List<List<Integer>> List, List<Integer> sub ){
   		 if(root==null) return;
   		 
   		 sub.add(root.val);
   		 //the case of reach the bottom leaf of tree
   		 if(sum==root.val && root.left==null && root.right==null){
   			 //Insert a clone of sub into List
   			 List.add(new ArrayList<Integer>(sub));
   		 }
   		 //Recursively through left and right sub tree
   		 helperDFS(root.left,sum-root.val,List,sub);
   		 helperDFS(root.right,sum-root.val,List,sub);
   		 //use Backtracking to deal with if next move is not fit, go back
   		 sub.remove(sub.size()-1);
   	}
       
   }
   ```

3. [Path Sum III](https://leetcode.com/problems/path-sum-iii) (437)  

   > Description

   You are given a binary tree in which each node contains an integer value.

   Find the number of paths that sum to a given value.

   The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

   The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

   **Example:**

   ```
   root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
   
         10
        /  \
       5   -3
      / \    \
     3   2   11
    / \   \
   3  -2   1
   
   Return 3. The paths that sum to 8 are:
   
   1.  5 -> 3
   2.  5 -> 2 -> 1
   3. -3 -> 11
   ```

   > Code_Java

   ```java
   /**
    * Definition for a binary tree node.
    * public class TreeNode {
    *     int val;
    *     TreeNode left;
    *     TreeNode right;
    *     TreeNode(int x) { val = x; }
    * }
    */
   class Solution {
       public int pathSum(TreeNode root, int sum) {
           if(root == null){
               return 0;
           }
           return dfs(root,sum)+pathSum(root.left,sum)+pathSum(root.right,sum);
           
       }
       private int dfs(TreeNode root,int sum){
           int res = 0;
           if(root == null){
               return res;
           }
           if(root.val ==sum){
               res++;
           }
           res+=dfs(root.left,sum-root.val);
           res+=dfs(root.right,sum-root.val);
           return res;
           
       }
   }
   
   // O(n) solution: https://leetcode.com/problems/path-sum-iii/discuss/91878/17-ms-O(n)-java-Prefix-sum-method 
   ```

###### 4.4 分解让复杂问题简单化

> 面试题35：复杂链表的复制

1. [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer) (138)

   > Description

   A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

   Return a deep copy of the list.

   > Code_Java

   ```java
   /**
    * Definition for singly-linked list with a random pointer.
    * class RandomListNode {
    *     int label;
    *     RandomListNode next, random;
    *     RandomListNode(int x) { this.label = x; }
    * };
    */
   public class Solution {
       public RandomListNode copyRandomList(RandomListNode head) {
           cloneNodes(head);
           connectSiblings(head);
           return separateList(head);
       }

       private void cloneNodes(RandomListNode head) {
           // First round: make copy of each node,
           // and link them together side-by-side in a single list.
           RandomListNode iter = head, next;

           while (iter != null) {
               next = iter.next;

               RandomListNode copy = new RandomListNode(iter.label);
               iter.next = copy;
               copy.next = next;

               iter = next;
           }
       }

       private void connectSiblings(RandomListNode head) {
           // Second round: assign random pointers for the copy nodes.
           RandomListNode iter = head;

           while (iter != null) {
               if (iter.random != null) {
                   iter.next.random = iter.random.next;
               }
               iter = iter.next.next; 
           }
       }

       // Third round: restore the original list, and extract the copy list.
       private RandomListNode separateList(RandomListNode head) {
           RandomListNode iter = head, next;
           RandomListNode dummyHead = new RandomListNode(-1);
           RandomListNode copy, copyIter = dummyHead;

           while (iter != null) {
               next = iter.next.next;

               // extract the copy
               copy = iter.next;
               copyIter.next = copy;
               copyIter = copy;

               // restore the original list
               iter.next = next;

               iter = next;
           }

           return dummyHead.next;
       }
   }
   // ref: https://leetcode.com/problems/copy-list-with-random-pointer/discuss/43491/A-solution-with-constant-space-complexity-O(1)-and-linear-time-complexity-O(N) 
   
   
      // hashmap: https://leetcode.com/problems/copy-list-with-random-pointer/discuss/43488/Java-O(n)-solution 
      public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) return null;

        Map<RandomListNode, RandomListNode> map = new HashMap<RandomListNode, RandomListNode>();

        // loop 1. copy all the nodes
        RandomListNode node = head;
        while (node != null) {
          map.put(node, new RandomListNode(node.label));
          node = node.next;
        }

        // loop 2. assign next and random pointers
        node = head;
        while (node != null) {
          map.get(node).next = map.get(node.next);
          map.get(node).random = map.get(node.random);
          node = node.next;
        }

        return map.get(head);
      }
   ```

> 面试题36：二叉搜索树与双向链表

1. [Convert Binary Search Tree to Sorted Doubly Linked List](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)(426)

   > Description

   输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。 

   > Code_Java

   ```java
   /**
   public class TreeNode {
       int val = 0;
       TreeNode left = null;
       TreeNode right = null;
   
       public TreeNode(int val) {
           this.val = val;
   
       }
   
   }
   */
   class Solution {
       Node prev;
       public Node treeToDoublyList(Node root) {
           if (root == null) return null;
           Node dummy = new Node(-1, null, null);
           prev = dummy;

           helper(root);

           // connect head and tail
           prev.right = dummy.right;
           dummy.right.left = prev;

           return dummy.right;
       }

       private void helper(Node cur) {
           if (cur == null) return;

           helper(cur.left);

           prev.right = cur;
           cur.left = prev;
           prev = cur;

           helper(cur.right);
       }
   }
   
   // devide and conquer: https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/discuss/154659/Divide-and-Conquer-without-Dummy-Node-Java-Solution
   ```

> 面试题37：序列化二叉树

1. [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree) (297)

   > Description

   Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

   Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

   **Example:** 

   ```
   You may serialize the following tree:
   
       1
      / \
     2   3
        / \
       4   5
   
   as "[1,2,3,null,null,4,5]"
   ```

   **Clarification:** The above format is the same as [how LeetCode serializes a binary tree](https://leetcode.com/faq/#binary-tree). You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

   **Note:** Do not use class member/global/static variables to store states. Your serialize and deserialize algorithms should be stateless.

   > Code_Java

   ```java
   
   ```

> 面试题38：字符串的排列

1. [Permutations](https://leetcode.com/problems/permutations/)(46)

   > Description

   输入一个字符串，打印出该字符串中字符的所有排列。 

   例如输入字符串abc，则打印由字符a,b,c所能排列出来的所有字符串：abc，abc,bac,bca,cab,cba 

   > Code_Java

   ```java
   // https://leetcode.com/problems/permutations/discuss/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partioning)
   ```

   
