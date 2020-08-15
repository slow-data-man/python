# -*- coding: utf-8 -*-
# @Time    : 2020/8/9 15:21
# @Author  : wangxg
from data_structure.base.node import TreeNode


class BinaryTree:
    """
    二叉树基本知识：
    1、二叉树的生成add
    2、二叉树的遍历，分别是广度遍历（一层一层的遍历），深度遍历（先序、中序、后序）
    """
    def __init__(self, root=None):
        self.root = root

    def add(self, value):
        """二叉树添加元素"""
        node = TreeNode(value)
        if self.root is None:
            self.root = node
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            if cur_node.left is None:
                cur_node.left = node
                return
            else:
                queue.append(cur_node.left)
            if cur_node.right is None:
                cur_node.right = node
                return
            else:
                queue.append(cur_node.right)

    def breadth_travel(self):
        """广度遍历"""
        lst = []
        queqe = [self.root]
        while queqe:
            cur_node = queqe.pop(0)
            lst.append(cur_node.value)
            if cur_node.left:
                queqe.append(cur_node.left)
            if cur_node.right:
                queqe.append(cur_node.right)
        return lst

    def preorder(self, root):
        """深度遍历：先序遍历（根左右）"""
        if root is None:
            return
        print(root.value)
        self.preorder(root.left)
        self.preorder(root.right)

    def midorder(self, root):
        """深度遍历：中序遍历（左根右）"""
        if root is None:
            return
        self.midorder(root.left)
        print(root.value)
        self.midorder(root.right)

    def postorder(self, root):
        """深度遍历：后序遍历（左右根）"""
        if root is None:
            return
        self.postorder(root.left)
        self.postorder(root.right)
        print(root.value)


if __name__ == '__main__':
    tree = BinaryTree()
    for i in range(1, 10):
        tree.add(i)
    tree.postorder(tree.root)
